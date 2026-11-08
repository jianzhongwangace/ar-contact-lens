"""Standalone EllSeg inference, coordinate transforms and visualization."""
from pathlib import Path
import hashlib
import json
import time
import cv2
import numpy as np
import torch
if __package__:
    from .vendor.ellseg_model import EllSegNetwork
else:
    from vendor.ellseg_model import EllSegNetwork

ROOT=Path(__file__).resolve().parent
WEIGHTS=ROOT/'weights/all.git_ok'
INPUT_SIZE=(320,240)

def preprocess(image):
    if image is None or image.ndim not in (2,3) or min(image.shape[:2])<8:
        raise ValueError('Input must be a valid eye image of at least 8x8 pixels')
    gray=image if image.ndim==2 else cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    h,w=gray.shape
    scale=320/w
    # Upstream fits width then centers with vertical crop/pad. Keep a precise
    # affine inverse rather than integer slice bookkeeping (broken upstream).
    affine=np.array([[scale,0,0],[0,scale,(240-h*scale)/2]],np.float64)
    resized=cv2.warpAffine(gray,affine,INPUT_SIZE,flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_CONSTANT)
    x=resized.astype(np.float32)
    x=(x-x.mean())/max(float(x.std()),1e-6)
    return np.ascontiguousarray(x[None,None]),affine

def soft_center(logits):
    """Upstream get_seg2ptLoss: spatial softmax at temperature=4 on [-1,1]."""
    h,w=logits.shape
    flat=logits.astype(np.float64).reshape(-1)*4
    weights=np.exp(flat-flat.max());weights/=weights.sum()
    xx,yy=np.meshgrid(np.linspace(-1,1,w),np.linspace(-1,1,h))
    return np.array([weights@xx.ravel(),weights@yy.ravel()])

def transform_ellipse(params,affine):
    """Transform center + semi-axes + angle using covariance/SVD, not axis scaling."""
    params=np.asarray(params,np.float64)
    if not np.isfinite(params).all() or min(params[2:4])<=0:return None
    c,s=np.cos(params[4]),np.sin(params[4])
    shape=np.array([[c,-s],[s,c]])@np.diag(params[2:4])
    transformed=affine[:,:2]@shape
    u,axes,_=np.linalg.svd(transformed)
    center=affine[:,:2]@params[:2]+affine[:,2]
    return np.array([*center,*axes,float(np.arctan2(u[1,0],u[0,0]))])

def decode(logits,regression,affine,shape):
    mask=logits.argmax(0).astype(np.uint8)
    inv=cv2.invertAffineTransform(affine)
    restored=cv2.warpAffine(mask,inv,(shape[1],shape[0]),flags=cv2.INTER_NEAREST)
    # Official runtime combines soft segmentation centers with these regression
    # indices. Names of temporary variables in upstream utils are misleading.
    normalization=np.array([[160,0,160],[0,120,120]],np.float64)
    normalized_to_source=(np.vstack([inv,[0,0,1]])@np.vstack([normalization,[0,0,1]]))[:2]
    ellipses={}
    for name,center,axes in [('iris',soft_center(-logits[0]),regression[2:5]),
                             ('pupil',soft_center(logits[2]),regression[7:10])]:
        ellipse=transform_ellipse(np.r_[center,axes],normalized_to_source)
        label_mask=(mask>0) if name=='iris' else (mask==2)
        # Merely a numerical/output plausibility gate, not calibrated confidence.
        credible=bool(ellipse is not None and label_mask.sum()>=20 and
                      0<=ellipse[0]<shape[1] and 0<=ellipse[1]<shape[0] and
                      min(ellipse[2:4])>=1 and max(ellipse[2:4])<max(shape[:2]))
        ellipses[name]={'ellipse':None if ellipse is None else ellipse.tolist(),
                        'plausible':credible,'segmentation_pixels':int(label_mask.sum())}
    return restored,ellipses

class EllSegPredictor:
    def __init__(self,device='auto',threads=4):
        torch.set_num_threads(max(1,threads))
        self.model=None;self.session=None;self.adapter=None
        expected=json.loads((ROOT/'vendor/source.json').read_text())['checkpoint_sha256']
        if hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()!=expected:
            raise RuntimeError('EllSeg checkpoint SHA256 mismatch')
        if device=='auto':
            device='cpu'
            if (ROOT/'weights/ellseg.onnx').exists():
                try:
                    import onnxruntime as ort
                    if 'DmlExecutionProvider' in ort.get_available_providers():device='dml'
                except ImportError:pass
        self.device=device
        if device in ('dml','onnx-cpu'):
            import onnxruntime as ort
            path=ROOT/'weights/ellseg.onnx'
            if not path.exists():raise FileNotFoundError('Run export_onnx.py first')
            manifest=json.loads(path.with_suffix('.json').read_text())
            if manifest['checkpoint_sha256']!=expected or hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['onnx_sha256']:
                raise RuntimeError('ONNX checksum/source mismatch')
            options=ort.SessionOptions();options.enable_mem_pattern=False
            options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
            options.intra_op_num_threads=max(1,threads)
            if device=='dml':
                if 'DmlExecutionProvider' not in ort.get_available_providers():raise RuntimeError('DirectML not installed')
                from gpu_info import preferred_adapter
                self.adapter=preferred_adapter()
                providers=[('DmlExecutionProvider',{'device_id':str(self.adapter['index'])}),'CPUExecutionProvider']
            else:providers=['CPUExecutionProvider']
            self.session=ort.InferenceSession(str(path),sess_options=options,providers=providers)
            self.session.disable_fallback()
            if device=='dml' and self.session.get_providers()[0]!='DmlExecutionProvider':raise RuntimeError('DirectML session failed')
        else:
            if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; use cpu or dml')
            self.model=EllSegNetwork()
            weights=torch.load(WEIGHTS,map_location='cpu',weights_only=True)['state_dict']
            self.model.load_state_dict(weights,strict=True)
            self.model.eval().to(device)

    def run(self,x):
        if self.session:return self.session.run(['logits','regression'],{'eye':x})
        with torch.inference_mode():
            logits,regression=self.model(torch.from_numpy(x).to(self.device))
        return logits.cpu().numpy(),regression.cpu().numpy()

    def predict(self,image):
        start=time.perf_counter()
        x,affine=preprocess(image)
        logits,regression=self.run(x)
        if not np.isfinite(logits).all() or not np.isfinite(regression).all():raise RuntimeError('Non-finite model output')
        mask,ellipses=decode(logits[0],regression[0],affine,image.shape)
        gray=image if image.ndim==2 else cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        contrast=float(np.std(gray))
        # Flat/dark missing frames are not eyes. InstanceNorm makes their tiny
        # rounding differences unstable; keep outputs diagnostic, mark invalid.
        input_valid=contrast>=2.0
        if not input_valid:
            mask[:]=0
            for record in ellipses.values():record['plausible']=False
        return {'mask':mask,'ellipses':ellipses,'processing_ms':(time.perf_counter()-start)*1000,
                'preprocess_affine':affine.tolist(),'input_valid':input_valid,'input_std':contrast}

def annotate(image,result):
    view=image.copy()
    for label,color in [(1,[75,180,75]),(2,[0,170,255])]:
        selected=result['mask']==label
        view[selected]=(.65*view[selected]+.35*np.array(color)).astype(np.uint8)
    for name,color in [('iris',(80,255,80)),('pupil',(0,190,255))]:
        record=result['ellipses'][name]
        if record['ellipse'] is None or not record['plausible']:continue
        cx,cy,a,b,angle=record['ellipse']
        cv2.ellipse(view,((cx,cy),(2*a,2*b),np.rad2deg(angle)),color,1,cv2.LINE_AA)
        cv2.drawMarker(view,(round(cx),round(cy)),color,cv2.MARKER_CROSS,7,1)
    return view
