"""Supplemental EllSeg geometry for AR. Inferred full ellipses are NOT visibility masks."""
import importlib
import sys
import types
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent

def verified_ellseg_support(eye,result,ellipse):
    """Require an independent LightIris pupil, not just EllSeg's own ellipse."""
    mask=result.get('mask')
    if mask is None or mask.shape!=eye.eyelid.shape:return None
    # Deferred import avoids a module startup cycle with the renderer.
    from lens_renderer import estimate_pupil
    cx,cy,a,b,theta=ellipse
    pupil=estimate_pupil(eye,np.array([cx,cy]),a,b)
    if pupil is None or eye.pupil_prob is None:return None
    yy,xx=np.indices(mask.shape);cs,sn=np.cos(theta),np.sin(theta)
    radius=(((xx-cx)*cs+(yy-cy)*sn)/a)**2+((-(xx-cx)*sn+(yy-cy)*cs)/b)**2
    visible=(radius<=1)&(eye.eyelid>0)
    support=(mask>0)&(eye.eyelid>0)
    ring=visible&(radius>.45**2)
    if support.sum()<12 or np.sum(support&visible)/max(1,support.sum())<.75:return None
    if np.sum(support&visible)/max(1,visible.sum())<.35:return None
    if np.sum(support&ring)/max(1,ring.sum())<.25:return None
    independent=(eye.pupil_prob>=.60)&(eye.eyelid>0)
    if np.sum(independent&(mask==2))/max(1,independent.sum())<.60:return None
    return support


def load_core():
    name='_ar_ellseg_deployment'
    if name not in sys.modules:
        package=types.ModuleType(name)
        package.__path__=[str(ROOT/'EllSeg部署')]
        sys.modules[name]=package
    return importlib.import_module(name+'.ellseg_core')

def validate_geometry(eye,result):
    """Cross-check the learned full ellipse against independent landmarks/evidence."""
    if not result['input_valid'] or not eye.is_open:return None,'closed/invalid input'
    record=result['ellipses']['iris']
    if not record['plausible'] or record['ellipse'] is None:return None,'ellipse implausible'
    ellipse=np.asarray(record['ellipse'],np.float64).copy()
    if not np.isfinite(ellipse).all():return None,'nonfinite ellipse'
    c=eye.center
    reference=np.ptp(eye.iris_points,axis=0)/2
    if np.min(reference)<2:return None,'landmarks too small'
    cx,cy,a,b,theta=ellipse
    cs,sn=np.cos(theta),np.sin(theta)
    extents=np.sqrt([(a*cs)**2+(b*sn)**2,(a*sn)**2+(b*cs)**2])
    ratios=extents/reference
    if np.any(ratios<.65) or np.any(ratios>1.65):return None,'size disagreement'
    if np.linalg.norm((ellipse[:2]-c)/reference)>.65:return None,'center disagreement'
    if max(a,b)/min(a,b)>2.8:return None,'extreme ellipse'
    x,y=np.rint(ellipse[:2]).astype(int)
    h,w=eye.eyelid.shape
    if not (0<=x<w and 0<=y<h and eye.eyelid[y,x]):return None,'center outside eyelids'
    mask=np.zeros((h,w),np.uint8)
    cv2.ellipse(mask,((cx,cy),(2*a,2*b),np.rad2deg(theta)),1,-1)
    visible=(mask>0)&(eye.eyelid>0)
    evidence=eye.iris>0
    reason='accepted'
    if evidence.sum()<12 or np.sum(evidence&visible)/max(1,evidence.sum())<.60:
        reason='visible iris disagreement'
    elif np.sum(evidence&visible)/max(1,visible.sum())<.12:reason='too little visible evidence'
    if reason!='accepted':
        if verified_ellseg_support(eye,result,ellipse) is None:return None,reason
        reason='accepted with independent pupil'
    # Represent major/minor consistently with the crop's horizontal iris axis.
    theta=(theta+np.pi/2)%np.pi-np.pi/2
    if abs(theta)>np.pi/4:
        a,b=b,a;theta=theta-np.sign(theta)*np.pi/2
    if abs(a-b)/max(a,b)<.08:theta=0. # Near-circle orientation is undefined/noisy.
    return np.array([cx,cy,a,b,theta]),reason

class EllSegGeometry:
    def __init__(self,device='auto',threads=4):
        self.core=load_core()
        self.predictor=self.core.EllSegPredictor(device,threads)
        self.device=self.predictor.device

    def enrich(self,eye):
        result=self.predictor.predict(eye.crop)
        eye.ellseg_result=result
        ellipse,reason=validate_geometry(eye,result)
        eye.geometry_reason=reason
        eye.geometry_ellipse=ellipse
        eye.geometry_source='EllSeg' if ellipse is not None else 'FaceMesh fallback'
        return ellipse is not None
