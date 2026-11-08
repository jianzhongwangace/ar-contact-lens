"""Current-image verification of short gaps, not a blind last-position hold."""
import cv2
import numpy as np

def verify_eye_motion(previous,eye):
    old=previous.get('flow_gray');mask=previous.get('flow_mask')
    if old is None or mask is None or old.shape!=eye.crop.shape[:2]:return False,{}
    gray=cv2.cvtColor(eye.crop,cv2.COLOR_BGR2GRAY)
    if float(gray.std())<4 or float(old.std())<4:return False,{}
    points=cv2.goodFeaturesToTrack(old,maxCorners=48,qualityLevel=.015,minDistance=3,
                                  mask=mask.astype(np.uint8)*255,blockSize=3)
    if points is None or len(points)<6:return False,{}
    current,status,_=cv2.calcOpticalFlowPyrLK(old,gray,points,None,winSize=(15,15),maxLevel=2)
    if current is None:return False,{}
    back,back_status,_=cv2.calcOpticalFlowPyrLK(gray,old,current,None,winSize=(15,15),maxLevel=2)
    if back is None:return False,{}
    error=np.linalg.norm(points[:,0]-back[:,0],axis=1)
    valid=(status[:,0]>0)&(back_status[:,0]>0)&(error<.8)
    if valid.sum()<6 or valid.mean()<.55:return False,{'flow_points':int(valid.sum())}
    transform,inliers=cv2.estimateAffinePartial2D(points[valid],current[valid],method=cv2.RANSAC,
                                                 ransacReprojThreshold=1.5,maxIters=500)
    if transform is None or inliers.mean()<.7:return False,{}
    zoom=float(np.linalg.norm(transform[:,0]))
    if not .8<zoom<1.22:return False,{}
    transported=transform[:,:2]@previous['flow_center']+transform[:,2]
    radius=max(2,float(np.mean(np.ptp(eye.iris_points,axis=0)/2)))
    if np.linalg.norm(transported-eye.center)>max(2.,radius*.35):return False,{}
    # Independent appearance check prevents tracks on a hand/eyelid/background
    # from extending an already unreliable segmentation sequence.
    warped=cv2.warpAffine(old,transform,(gray.shape[1],gray.shape[0]),flags=cv2.INTER_LINEAR)
    region=cv2.warpAffine(mask.astype(np.uint8),transform,(gray.shape[1],gray.shape[0]),flags=cv2.INTER_NEAREST)>0
    region&=eye.eyelid>0
    if region.sum()<30:return False,{}
    a=warped[region].astype(np.float32);b=gray[region].astype(np.float32)
    if a.std()<3 or b.std()<3:return False,{}
    correlation=float(np.mean((a-a.mean())*(b-b.mean()))/(a.std()*b.std()))
    return correlation>.80,{'flow_points':int(valid.sum()),'flow_correlation':correlation}
