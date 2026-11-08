"""Same-frame FaceMesh -> LightIrisNet -> eyelid visibility pipeline."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time
import warnings

import cv2
import numpy as np
import torch

from vendor.lightiris_model import IrisNetDeepLab
from mediapipe_compat import load_mediapipe

mp = load_mediapipe()

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / 'research/LightIrisNet/model/best_mobilenetv3.pt'
EYES = {
    'right': {
        'contour': [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7],
        'corners': [33,133], 'vertical': [(160,144),(159,145),(158,153)], 'iris': 468,
    },
    'left': {
        'contour': [362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382],
        'corners': [362,263], 'vertical': [(385,380),(386,374),(387,373)], 'iris': 473,
    },
}


def component_near(mask, center):
    """Keep the region containing the FaceMesh iris center, else nearest region."""
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if count <= 1:
        return np.zeros_like(mask, dtype=np.uint8)
    x, y = np.rint(center).astype(int)
    label = int(labels[y, x]) if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1] else 0
    if label == 0:
        candidates = [i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA] >= 3]
        if not candidates:
            return np.zeros_like(mask, dtype=np.uint8)
        label = min(candidates, key=lambda i: float(np.linalg.norm(centroids[i]-center)))
    return (labels == label).astype(np.uint8)


def visible_masks(iris_prob, pupil_prob, eyelid, center, is_open, threshold=0.5):
    """Geometry clips AI predictions; no synthetic iris substitutes on failures."""
    if not is_open:
        empty = np.zeros_like(eyelid, dtype=np.uint8)
        return empty, empty.copy()
    iris = component_near((iris_prob >= threshold) & (eyelid > 0), center)
    pupil = component_near((pupil_prob >= threshold) & (iris > 0), center)
    return iris, pupil


@dataclass
class EyeResult:
    side: str
    crop: np.ndarray
    affine: np.ndarray
    contour: np.ndarray
    center: np.ndarray
    eyelid: np.ndarray
    openness: float
    is_open: bool
    iris_prob: np.ndarray | None = None
    pupil_prob: np.ndarray | None = None
    iris: np.ndarray | None = None
    pupil: np.ndarray | None = None
    iris_points: np.ndarray | None = None
    lens_pose: np.ndarray | None = None
    lens_opacity: float = 1.
    tracking_state: str = 'unreliable'
    pupil_protection: np.ndarray | None = None
    lens_support: np.ndarray | None = None
    display_affine: np.ndarray | None = None
    pupil_anchor: np.ndarray | None = None
    pupil_observed: np.ndarray | None = None
    pupil_confidence: float = 0.
    pupil_source: str = 'iris fallback'
    geometry_ellipse: np.ndarray | None = None
    geometry_source: str = 'FaceMesh'
    geometry_reason: str = ''
    ellseg_result: dict | None = None
    tracking_reason: str = ''


class LightIrisPredictor:
    def __init__(self, weights=DEFAULT_WEIGHTS, device='auto', threads=4):
        torch.set_num_threads(max(1,threads))
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable. Use --device cpu or install a compatible CUDA PyTorch build.')
        self.device = torch.device(device)
        weights = Path(weights).resolve()
        if not weights.is_file():
            raise FileNotFoundError(f'LightIrisNet checkpoint missing: {weights}')
        source = json.loads((ROOT/'vendor/model_source.json').read_text())
        if weights == DEFAULT_WEIGHTS.resolve():
            actual = hashlib.sha256(weights.read_bytes()).hexdigest()
            if actual != source['sha256']:
                raise RuntimeError('Checkpoint checksum mismatch; restore the original supplied weights.')
        checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
        state = checkpoint.get('state_dict', checkpoint)
        self.model = IrisNetDeepLab(
            backbone='mobilenetv3', use_ellipse=True,
            extra_decoder_conv=True, pupil_refine_depth=1,
        )
        # A successful strict load proves every expected layer has supplied weights.
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(self.device)
        self.mean = torch.tensor([.485,.456,.406],device=self.device).view(1,3,1,1)
        self.std = torch.tensor([.229,.224,.225],device=self.device).view(1,3,1,1)
        self.parameters = sum(p.numel() for p in self.model.parameters())

    @torch.inference_mode()
    def predict(self, crops):
        rgb = np.stack([cv2.cvtColor(x,cv2.COLOR_BGR2RGB) for x in crops])
        x = torch.from_numpy(rgb.transpose(0,3,1,2).copy()).to(self.device,dtype=torch.float32)/255.0
        x = (x-self.mean)/self.std
        iris,pupil = self.segmentation_logits(x)
        return (iris.sigmoid().cpu().numpy()[:,0],
                pupil.sigmoid().cpu().numpy()[:,0])

    def segmentation_logits(self, x):
        # Identical main-head operations to the upstream forward. Unused
        # boundary/distance/ellipse heads are omitted only at inference time.
        model = self.model
        features = model.enc(x)
        z = model.dec(model.aspp(features['s32']),features['s4'])
        z = torch.nn.functional.interpolate(z,size=x.shape[-2:],mode='bilinear',align_corners=False)
        return model.head_iris(z), model.head_pupil(model.pre_pupil(z))


class EyePipeline:
    def __init__(self, weights=DEFAULT_WEIGHTS, device='auto', input_width=256,
                 input_height=160, threshold=.5, close_threshold=.10, threads=4,
                 static_image=False,geometry='facemesh'):
        if input_width < 64 or input_height < 64 or input_width % 32 or input_height % 32:
            raise ValueError('Input dimensions must be multiples of 32 and at least 64.')
        if not hasattr(mp,'solutions'):
            raise RuntimeError('This prototype requires mediapipe==0.10.21. Use the supplied launcher/environment.')
        if device=='auto' and Path(weights).resolve()==DEFAULT_WEIGHTS.resolve():
            onnx_path=ROOT/'models'/f'lightiris-2x{input_height}x{input_width}.onnx'
            if onnx_path.exists():
                try:
                    import onnxruntime as ort
                    if 'DmlExecutionProvider' in ort.get_available_providers():
                        from onnx_predictor import OnnxIrisPredictor
                        self.predictor=OnnxIrisPredictor(weights,input_width,input_height,'dml',threads)
                except Exception as exc:
                    warnings.warn(f'DirectML unavailable; falling back to PyTorch: {exc}')
        if hasattr(self,'predictor'):
            pass
        elif device in ('dml','onnx-cpu'):
            from onnx_predictor import OnnxIrisPredictor
            self.predictor=OnnxIrisPredictor(weights,input_width,input_height,device,threads)
        else:
            self.predictor = LightIrisPredictor(weights,device,threads)
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=.5, min_tracking_confidence=.5,
        )
        self.size = (input_width,input_height)
        self.threshold = threshold
        self.close_threshold = close_threshold
        self.open_states = {k:True for k in EYES}
        from lens_renderer import LensTracker,EyeViewTracker
        self.lens_tracker=LensTracker(static=static_image)
        self.eye_view_tracker=EyeViewTracker(static=static_image)
        self.geometry_mode='facemesh'
        self.ellseg_geometry=None
        self._geometry_device=device
        self._geometry_threads=threads
        self.set_geometry(geometry)

    def set_geometry(self,mode):
        if mode not in ('facemesh','ellseg'):raise ValueError('Unknown geometry mode')
        if mode=='ellseg' and self.ellseg_geometry is None:
            from ellseg_adapter import EllSegGeometry
            candidate=EllSegGeometry(self._geometry_device,self._geometry_threads)
            self.ellseg_geometry=candidate
        if mode!=self.geometry_mode:
            self.geometry_mode=mode
            self.reset_tracking()

    def close(self):
        self.mesh.close()

    def reset_tracking(self):
        """Discard state on video replacement, replay or seeking."""
        self.mesh.reset()
        self.open_states={k:True for k in EYES}
        self.lens_tracker.reset()
        self.eye_view_tracker.reset()

    def process(self, frame, timestamp=None):
        """All outputs refer to this frame, not a newer camera preview."""
        started = time.perf_counter()
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        detection = self.mesh.process(rgb)
        mesh_ms = (time.perf_counter()-started)*1000
        eyes = []
        if not detection.multi_face_landmarks:
            self.open_states = {k:True for k in EYES}
            self.lens_tracker.reset()
            self.eye_view_tracker.reset()
            return eyes, {'mesh_ms':mesh_ms,'iris_ms':0.,'total_ms':(time.perf_counter()-started)*1000}
        h,w = frame.shape[:2]
        pts = np.array([[p.x*w,p.y*h] for p in detection.multi_face_landmarks[0].landmark],dtype=np.float32)
        for side,spec in EYES.items():
            corners = pts[spec['corners']]
            a,b = sorted(corners,key=lambda p:p[0])
            width = float(np.linalg.norm(b-a))
            if width < 12:
                continue
            ux = (b-a)/width
            uy = np.array([-ux[1],ux[0]],dtype=np.float32)
            center = (a+b)/2
            crop_width = width*1.7
            crop_height = crop_width*self.size[1]/self.size[0]
            origin = center-ux*crop_width/2-uy*crop_height/2
            src = np.array([origin,origin+ux*crop_width,origin+uy*crop_height],dtype=np.float32)
            dst = np.array([[0,0],[self.size[0]-1,0],[0,self.size[1]-1]],dtype=np.float32)
            matrix = cv2.getAffineTransform(src,dst)
            crop = cv2.warpAffine(frame,matrix,self.size,borderMode=cv2.BORDER_REFLECT_101)
            contour = cv2.transform(pts[spec['contour']][None],matrix)[0]
            iris_center = cv2.transform(pts[[spec['iris']]][None],matrix)[0,0]
            mask = np.zeros((self.size[1],self.size[0]),dtype=np.uint8)
            cv2.fillPoly(mask,[np.rint(contour).astype(np.int32)],1)
            openness = float(np.mean([np.linalg.norm(pts[u]-pts[l]) for u,l in spec['vertical']])/width)
            # Hysteresis handles noise around the closed-eye threshold.
            cutoff = self.close_threshold if self.open_states[side] else self.close_threshold*1.25
            opened = openness >= cutoff
            self.open_states[side] = opened
            eye=EyeResult(side,crop,matrix,contour,iris_center,mask,openness,opened)
            eye.iris_points=cv2.transform(pts[spec['iris']+1:spec['iris']+5][None],matrix)[0]
            eyes.append(eye)
        infer_start = time.perf_counter()
        if eyes:
            iris_probs,pupil_probs = self.predictor.predict([e.crop for e in eyes])
            for e,ip,pp in zip(eyes,iris_probs,pupil_probs):
                e.iris_prob,e.pupil_prob = ip,pp
                e.iris,e.pupil = visible_masks(ip,pp,e.eyelid,e.center,e.is_open,self.threshold)
        ellseg_start=time.perf_counter()
        if self.geometry_mode=='ellseg':
            for eye in eyes:
                if eye.is_open:self.ellseg_geometry.enrich(eye)
                else:eye.geometry_source='closed';eye.geometry_reason='closed eye'
        ellseg_ms=(time.perf_counter()-ellseg_start)*1000
        self.lens_tracker.update(eyes,now=timestamp)
        self.eye_view_tracker.update(eyes,now=timestamp)
        return eyes, {'mesh_ms':mesh_ms,'iris_ms':(ellseg_start-infer_start)*1000,'ellseg_ms':ellseg_ms,
                      'total_ms':(time.perf_counter()-started)*1000}


def eye_overlay(e):
    view = e.crop.copy()
    ring = (e.iris > 0) & (e.pupil == 0)
    view[ring] = (.5*view[ring]+.5*np.array([80,220,70])).astype(np.uint8)
    p = e.pupil > 0
    view[p] = (.4*view[p]+.6*np.array([0,170,255])).astype(np.uint8)
    cv2.polylines(view,[e.contour.astype(np.int32)],True,(255,220,0),1,cv2.LINE_AA)
    if e.geometry_ellipse is not None:
        cx,cy,a,b,angle=e.geometry_ellipse
        cv2.ellipse(view,((cx,cy),(2*a,2*b),np.rad2deg(angle)),(255,120,220),1,cv2.LINE_AA)
    return view


def blend_eye_region(main, e, frame_to_target=None):
    """Warp masks only into the bounded eye ROI, preserving original camera pixels."""
    h,w=main.shape[:2]
    inverse=cv2.invertAffineTransform(e.affine)
    if frame_to_target is not None:
        inverse=(frame_to_target @ np.vstack([inverse,[0,0,1]]))[:2]
    ch,cw=e.iris.shape
    corners=np.array([[[0,0],[cw,0],[cw,ch],[0,ch]]],dtype=np.float32)
    projected=cv2.transform(corners,inverse)[0]
    x0,y0=np.maximum(np.floor(projected.min(axis=0)).astype(int)-2,[0,0])
    x1,y1=np.minimum(np.ceil(projected.max(axis=0)).astype(int)+2,[w,h])
    if x1<=x0 or y1<=y0:return
    local=inverse.copy()
    local[:,2]-=[x0,y0]
    # A packed two-channel warp replaces two full-frame warps.
    masks=cv2.warpAffine(np.stack([e.iris,e.pupil],axis=-1),local,(x1-x0,y1-y0),flags=cv2.INTER_NEAREST)
    iris=masks[:,:,0]>0
    pupil=masks[:,:,1]>0
    region=main[y0:y1,x0:x1]
    ring=iris & ~pupil
    region[ring]=(.5*region[ring]+.5*np.array([80,220,70])).astype(np.uint8)
    region[pupil]=(.4*region[pupil]+.6*np.array([0,170,255])).astype(np.uint8)
    contour=cv2.transform(e.contour[None],inverse)[0].astype(np.int32)
    cv2.polylines(main,[contour],True,(255,220,0),1,cv2.LINE_AA)


def final_eye_view(eye,material=None,strength=.70,lens_scale=1.,overlay=True,mirror=False,source_result=None):
    """Crop the actual final frame, so detail views never invent a different effect."""
    display_affine=eye.display_affine if eye.display_affine is not None else eye.affine
    view=(cv2.warpAffine(source_result,display_affine,(eye.crop.shape[1],eye.crop.shape[0]),
                        borderMode=cv2.BORDER_REFLECT_101) if source_result is not None else eye.crop.copy())
    if source_result is None and overlay and material is not None:
        from lens_renderer import compose_lens
        compose_lens(view,eye,material,strength,lens_scale,
                     np.vstack([eye.affine,[0,0,1]]))
    if mirror:view=cv2.flip(view,1)
    return view


def _paste_fit(canvas,image,box):
    x,y,w,h=box
    ih,iw=image.shape[:2]
    factor=min(w/iw,h/ih)
    dw,dh=max(1,round(iw*factor)),max(1,round(ih*factor))
    resized=cv2.resize(image,(dw,dh),interpolation=cv2.INTER_LINEAR)
    canvas[y+(h-dh)//2:y+(h-dh)//2+dh,x+(w-dw)//2:x+(w-dw)//2+dw]=resized


def alignment_view(eye):
    view=eye.crop.copy()
    cv2.drawMarker(view,tuple(np.rint(eye.center).astype(int)),(255,200,0),cv2.MARKER_CROSS,9,1)
    if eye.pupil_observed is not None:
        point=cv2.transform(np.asarray(eye.pupil_observed,np.float32)[None,None],eye.affine)[0,0]
        cv2.circle(view,tuple(np.rint(point).astype(int)),4,(0,140,255),1)
    if eye.pupil_anchor is not None:
        point=cv2.transform(np.asarray(eye.pupil_anchor,np.float32)[None,None],eye.affine)[0,0]
        cv2.drawMarker(view,tuple(np.rint(point).astype(int)),(60,255,60),cv2.MARKER_CROSS,9,1)
    return view


def render(frame, eyes, timing, fps, mirror=True, overlay=True, image_mode=False,
           material=None,strength=.70,lens_scale=1.,video_mode=False,alignment_debug=False):
    from app_layout import WIDTH,CONTENT_HEIGHT
    from PIL import Image,ImageDraw,ImageFont
    canvas=np.full((CONTENT_HEIGHT,WIDTH,3),24,np.uint8)
    # Static Chinese labels cached to avoid re-rasterizing fonts on video frames.
    if not hasattr(render,'labels'):
        label_image=Image.new('RGB',(WIDTH,CONTENT_HEIGHT),(24,24,24))
        draw=ImageDraw.Draw(label_image)
        font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21)
        small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
        for xy,text in [((18,15),'双眼佩戴效果 · 局部放大'),((390,15),'完整画面'),((1395,15),'识别诊断')]:
            draw.text(xy,text,font=font,fill=(235,235,235))
        draw.text((18,620),'放大当前眼部画面，不额外生成细节',font=small,fill=(170,180,190))
        draw.text((18,648),'O 原眼对比 · M 镜像切换',font=small,fill=(190,200,215))
        draw.line((365,0,365,CONTENT_HEIGHT),fill=(55,60,68))
        draw.line((1380,0,1380,CONTENT_HEIGHT),fill=(55,60,68))
        render.labels=cv2.cvtColor(np.asarray(label_image),cv2.COLOR_RGB2BGR)
    canvas[:]=render.labels
    h,w=frame.shape[:2]
    scale=min(980/w,638/h)
    dw,dh=max(1,round(w*scale)),max(1,round(h*scale))
    final_frame=frame
    if overlay and material is not None:
        from lens_renderer import compose_lens
        final_frame=frame.copy()
        for eye in eyes:compose_lens(final_frame,eye,material,strength,lens_scale)
    main=cv2.resize(final_frame,(dw,dh))
    sx,sy=dw/w,dh/h
    transform=np.array([[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1]],dtype=np.float64)
    if overlay and material is None:
        for e in eyes:
            blend_eye_region(main,e,transform)
    if mirror:main=cv2.flip(main,1)
    x,y=385+(980-dw)//2,50+(638-dh)//2
    canvas[y:y+dh,x:x+dw]=main
    rate='Photo' if image_mode else f'{fps:.1f} processed FPS'
    cv2.putText(canvas,f'{rate} | mesh {timing["mesh_ms"]:.0f} ms | iris {timing["iris_ms"]:.0f} ms | EllSeg {timing.get("ellseg_ms",0):.0f} ms',
                (390,722),cv2.FONT_HERSHEY_SIMPLEX,.55,(220,220,220),1,cv2.LINE_AA)
    for i,side in enumerate(('right','left')):
        eye=next((e for e in eyes if e.side==side),None)
        top=58+i*270
        cv2.putText(canvas,f'Your {side} eye',(18,top+22),cv2.FONT_HERSHEY_SIMPLEX,.65,(235,235,235),1,cv2.LINE_AA)
        status='No eye detected'
        if eye is not None:
            status=('Original' if not overlay else ('Import a lens image' if material is None else
                    ('Closed - lens hidden' if not eye.is_open else
                     (('Try-on: '+eye.tracking_state) if eye.lens_pose is not None else 'Tracking uncertain - hidden'))))
            closeup=final_eye_view(eye,mirror=mirror,source_result=final_frame)
            _paste_fit(canvas,closeup,(16,top+40,332,195))
        else:
            cv2.rectangle(canvas,(16,top+40),(348,top+235),(33,35,39),-1)
        cv2.putText(canvas,status,(18,top+257),cv2.FONT_HERSHEY_SIMPLEX,.46,(190,210,220),1,cv2.LINE_AA)
        dy=60+i*230
        cv2.putText(canvas,f'Your {side} eye: '+('not detected' if eye is None else ('OPEN' if eye.is_open else 'CLOSED')),
                    (1395,dy+20),cv2.FONT_HERSHEY_SIMPLEX,.52,(230,230,230),1,cv2.LINE_AA)
        if eye is None:continue
        raw=np.zeros_like(eye.crop)
        raw[eye.iris_prob>=.5]=(80,220,70);raw[eye.pupil_prob>=.5]=(0,170,255)
        diagnostic=(alignment_view(eye),'Pupil / aperture') if alignment_debug else (eye_overlay(eye),'Visible mask')
        tiles=[(eye.crop,'Input crop'),(raw,'Raw AI >= 0.5'),diagnostic]
        for j,(tile,label) in enumerate(tiles):
            tx=1390+j*175
            cv2.putText(canvas,label,(tx,dy+48),cv2.FONT_HERSHEY_SIMPLEX,.4,(200,200,200),1,cv2.LINE_AA)
            _paste_fit(canvas,tile,(tx,dy+60,165,115))
        source_width=(eye.crop.shape[1]-1)/np.linalg.norm(eye.affine[0,:2])/1.7
        info=f'{eye.geometry_source} | eye={source_width:.0f}px | opening={eye.openness:.2f}'
        if alignment_debug:
            error=(np.linalg.norm(eye.pupil_observed-eye.pupil_anchor)
                   if eye.pupil_observed is not None and eye.pupil_anchor is not None else None)
            info=(f'{eye.pupil_source} | q={eye.pupil_confidence:.2f} | error={error:.2f}px'
                  if error is not None else f'{eye.pupil_source} | no reliable pupil observation')
        cv2.putText(canvas,info,
                    (1395,dy+201),cv2.FONT_HERSHEY_SIMPLEX,.44,(190,190,190),1,cv2.LINE_AA)
        cv2.putText(canvas,eye.tracking_reason,(1395,dy+219),cv2.FONT_HERSHEY_SIMPLEX,.36,(180,185,205),1,cv2.LINE_AA)
    help_lines=[('Orange: measured pupil | Green: aperture' if alignment_debug else 'Green: iris | Orange: pupil | Cyan: eyelid'),
                'Q / Esc: quit     M: mirror preview',
                'O: compare | S: save | D: alignment',
                'L: choose another image' if image_mode else ('Space: pause | R: replay | V: video' if video_mode else 'V: open offline video')]
    for i,text in enumerate(help_lines):
        cv2.putText(canvas,text,(1395,570+i*26),cv2.FONT_HERSHEY_SIMPLEX,.43,(210,210,210),1,cv2.LINE_AA)
    hint='No face detected in this frame.' if not eyes else ('No reliable iris mask; lens is hidden.' if not any(e.iris.any() for e in eyes) else 'Check close-ups for actual lens alignment.')
    cv2.putText(canvas,hint,(1395,705),cv2.FONT_HERSHEY_SIMPLEX,.43,(100,210,255),1,cv2.LINE_AA)
    return canvas
