"""Same material renderer for local photos, live preview and full-size export."""
import time
import cv2
import numpy as np
from lens_material import srgb_to_linear,linear_to_srgb,smoothstep
from lens_settings import clamp_lens_scale
from center_tracking import track_center,pose_basis,concentric_uv
from turn_tracking import verify_eye_motion
from ellseg_adapter import verified_ellseg_support


def stabilize_pose(raw,prior,dt):
    """Median outliers + deadbands + adaptive, time-based filtering in source pixels."""
    raw=raw.copy()
    previous=prior['pose']
    raw[4]=previous[4]+(raw[4]-previous[4]+np.pi)%(2*np.pi)-np.pi
    observations=(prior.get('observations',[previous])+[raw])[-3:]
    measured=np.median(np.stack(observations),axis=0)
    stable=previous.copy()
    radius=max(float(np.mean(previous[2:4])),1.)
    distance=float(np.linalg.norm(measured[:2]-previous[:2]))
    position_deadband=max(.30,radius*.025)
    if distance>position_deadband:
        motion=float(smoothstep(.025,.14,distance/radius))
        tau=.18*(1-motion)+.012*motion
        stable[:2]+=(1-np.exp(-dt/tau))*(measured[:2]-previous[:2])
    # Axes are slower than translation; small noisy radius changes must not
    # make the texture "breathe". Real zoom/pose changes unlock the filter.
    for axis in (2,3):
        relative=abs(measured[axis]-previous[axis])/max(previous[axis],1.)
        if abs(measured[axis]-previous[axis])>max(.15,.025*previous[axis]):
            motion=float(smoothstep(.025,.13,relative))
            tau=.28*(1-motion)+.035*motion
            stable[axis]+=(1-np.exp(-dt/tau))*(measured[axis]-previous[axis])
    rotation=measured[4]-previous[4]
    if abs(rotation)>np.deg2rad(.8):
        motion=float(smoothstep(np.deg2rad(.8),np.deg2rad(5),abs(rotation)))
        tau=.22*(1-motion)+.02*motion
        stable[4]+=(1-np.exp(-dt/tau))*rotation
    return stable,observations


def stabilize_support(current,previous,dt):
    """Filter small boundary noise in iris coordinates; large occlusions cut immediately."""
    union=(current>.1)|(previous>.1)
    change=float(np.abs(current-previous)[union].mean()) if union.any() else 0.
    if change>.16:
        # A substantial new occlusion is not noise. Do not average old visible
        # pixels into the occluder; expansion can still settle over a few frames.
        gain=1-np.exp(-dt/.045)
        return np.minimum(current,previous+gain*(current-previous))
    gain=1-np.exp(-dt/.16)
    filtered=previous+gain*(current-previous)
    # Permit only one canonical pixel of edge uncertainty, not unrestricted
    # persistence outside the currently supported iris region.
    boundary=cv2.dilate(current,np.ones((3,3),np.uint8))
    return np.minimum(filtered,boundary).astype(np.float32)


def support_at_pose(normalized,shape,pose,source_to_crop):
    """Project all stabilized masks with exactly the same geometry as the texture."""
    cs,sn=np.cos(pose[4]),np.sin(pose[4])
    basis=np.array([[cs*pose[2]/63,-sn*pose[3]/63],
                    [sn*pose[2]/63,cs*pose[3]/63]],np.float64)
    transform=np.column_stack([basis,pose[:2]-basis@np.array([64.,64.])])
    transform=(np.vstack([source_to_crop,[0,0,1]])@np.vstack([transform,[0,0,1]]))[:2]
    return cv2.warpAffine(normalized,transform,(shape[1],shape[0]),flags=cv2.INTER_LINEAR)


class EyeViewTracker:
    """Stabilize only the magnified display framing; never alter the model's ROI."""
    def __init__(self,static=False):self.static=static;self.history={}

    def reset(self):self.history.clear()

    def update(self,eyes,now=None):
        now=time.monotonic() if now is None else now
        seen=set()
        for e in eyes:
            seen.add(e.side)
            h,w=e.crop.shape[:2]
            inverse=cv2.invertAffineTransform(e.affine)
            crop_center=np.array([(w-1)/2,(h-1)/2])
            center=inverse[:,:2]@crop_center+inverse[:,2]
            raw=np.array([*center,np.linalg.norm(inverse[:,0])*(w-1)/2,
                          np.linalg.norm(inverse[:,1])*(h-1)/2,
                          np.arctan2(inverse[1,0],inverse[0,0])],np.float64)
            pose=raw;observations=[raw.copy()]
            prior=self.history.get(e.side)
            if not self.static and prior:
                dt=now-prior['time']
                move=np.linalg.norm(raw[:2]-prior['pose'][:2])/max(1,prior['pose'][2])
                ratio=raw[2:4]/prior['pose'][2:4]
                if 0<dt<.18 and move<.8 and np.all((ratio>.65)&(ratio<1.5)):
                    pose,observations=stabilize_pose(raw,prior,dt)
            cs,sn=np.cos(pose[4]),np.sin(pose[4])
            basis=np.array([[cs*pose[2],-sn*pose[3]],[sn*pose[2],cs*pose[3]]])
            matrix=np.diag(crop_center)@np.linalg.inv(basis)
            e.display_affine=np.column_stack([matrix,crop_center-matrix@pose[:2]])
            self.history[e.side]={'pose':pose,'time':now,'observations':observations}
        for side in set(self.history)-seen:self.history.pop(side,None)


class LensTracker:
    """Full lens geometry with bounded fallback for transient segmentation loss."""
    HOLD_SECONDS=.22
    VERIFIED_HOLD_SECONDS=.45

    def __init__(self,static=False):
        self.static=static
        self.history={}

    def reset(self):self.history.clear()

    def update(self,eyes,now=None):
        now=time.monotonic() if now is None else now
        seen=set()
        for e in eyes:
            e.lens_pose=None
            e.lens_opacity=0.
            e.tracking_state='unreliable'
            e.pupil_protection=None
            e.lens_support=None
            e.tracking_reason='unreliable geometry'
            e.pupil_anchor=None
            e.pupil_observed=None
            e.pupil_confidence=0.
            e.pupil_source='iris fallback'
            seen.add(e.side)
            if not e.is_open or e.iris_points is None:
                self.history.pop(e.side,None)
                e.tracking_state='closed' if not e.is_open else 'unreliable'
                e.tracking_reason='closed' if not e.is_open else 'missing iris landmarks'
                continue
            center=e.center.astype(np.float64)
            points=e.iris_points
            rx=float(np.ptp(points[:,0]))/2
            ry=float(np.ptp(points[:,1]))/2
            geometry=getattr(e,'geometry_ellipse',None)
            geometry_angle=0.
            if geometry is not None:
                center=np.asarray(geometry[:2],np.float64)
                rx,ry=map(float,geometry[2:4]);geometry_angle=float(geometry[4])
            if not np.isfinite([*center,rx,ry]).all() or min(rx,ry)<2 or max(rx,ry)/min(rx,ry)>3.0:
                self.history.pop(e.side,None);continue
            yy,xx=np.indices(e.iris.shape)
            cs,sn=np.cos(geometry_angle),np.sin(geometry_angle)
            radius=(((xx-center[0])*cs+(yy-center[1])*sn)/rx)**2+((-(xx-center[0])*sn+(yy-center[1])*cs)/ry)**2
            visible=(radius<=1)&(e.eyelid>0)
            # Segmentation verifies the pose, but does not cut holes in the lens.
            support=e.iris>0
            coverage=float(np.sum(support&visible)/max(1,visible.sum()))
            agreement=float(np.sum(support&(radius<1.5))/max(1,support.sum()))
            visible_fraction=float(visible.sum()/(np.pi*rx*ry))
            inv=cv2.invertAffineTransform(e.affine)
            center_src=cv2.transform(center.astype(np.float32)[None,None],inv)[0,0]
            scale=float(np.linalg.norm(inv[:,0]))
            roll=float(np.arctan2(inv[1,0],inv[0,0]))+geometry_angle
            current=np.array([*center_src,rx*scale,ry*scale,roll],np.float64)
            prior=None if self.static else self.history.get(e.side)
            dt=now-prior['time'] if prior else float('inf')
            move=np.linalg.norm(current[:2]-prior['pose'][:2])/max(prior['pose'][2],1) if prior else float('inf')
            ratio=current[2:4]/prior['pose'][2:4] if prior else np.ones(2)
            consistent=prior is not None and 0<dt<.18 and move<.8 and np.all((ratio>.65)&(ratio<1.5))
            reliable=(support.sum()>=12 and coverage>=(.10 if consistent else .15) and agreement>=.50 and visible_fraction>.15)
            secondary=False
            if not reliable and geometry is not None and getattr(e,'ellseg_result',None) is not None and visible_fraction>.15:
                secondary=verified_ellseg_support(e,e.ellseg_result,geometry) is not None
                reliable=secondary
            flow_ok=False
            # Low inference FPS is not itself evidence of a lost eye. Longer
            # intervals need current pixel verification, not blind reuse.
            motion_candidate=(prior is not None and 0<dt<.35 and move<.8 and np.all((ratio>.65)&(ratio<1.5)))
            if not reliable and not self.static and motion_candidate and visible_fraction>.15 and now-prior['reliable_time']<self.VERIFIED_HOLD_SECONDS:
                flow_ok,_=verify_eye_motion(prior,e)
                if flow_ok:consistent=True
            if geometry is not None:
                # EllSeg predicts the complete ellipse; the CURRENT eyelids
                # clip it here and again during compositing, including blinks.
                envelope=((radius<=1.)&(e.eyelid>0)).astype(np.float32)
                envelope=np.clip(cv2.distanceTransform(envelope.astype(np.uint8),cv2.DIST_L2,3),0,1)
            else:envelope=iris_envelope(e,center,rx,ry)
            if reliable and not envelope.any():reliable=False
            if reliable:
                last_reliable=now
                opacity=1.
                e.tracking_state='tracked'
                e.tracking_reason='EllSeg mask + independent pupil' if secondary else 'current segmentation'
            elif flow_ok:
                last_reliable=prior['reliable_time']
                age=now-last_reliable
                opacity=float(1-smoothstep(.25,self.VERIFIED_HOLD_SECONDS,age))
                e.tracking_state='verified motion'
                e.tracking_reason='current texture and iris points agree'
            elif not self.static and consistent and visible_fraction>.15 and now-prior['reliable_time']<self.HOLD_SECONDS:
                last_reliable=prior['reliable_time']
                age=now-last_reliable
                opacity=float(1-smoothstep(.08,self.HOLD_SECONDS,age))
                e.tracking_state='brief fallback'
                e.tracking_reason='short segmentation gap'
            else:
                e.tracking_reason=('no prior verification' if prior is None else
                                   'motion discontinuity' if not consistent else
                                   'too little eye visible' if visible_fraction<=.15 else 'segmentation gap expired')
                self.history.pop(e.side,None);continue
            observations=[current.copy()]
            raw_iris=current.copy()
            pupil_params=estimate_pupil(e,center,rx,ry)
            raw_pupil=None
            if pupil_params is not None:
                crop_observed=center+pupil_params[:2]*[rx,ry]
                raw_pupil=cv2.transform(crop_observed.astype(np.float32)[None,None],inv)[0,0].astype(np.float64)
            pupil_hint=(raw_pupil-prior['pupil_filter']['raw'] if raw_pupil is not None and consistent
                        and prior.get('pupil_filter') is not None else None)
            if not self.static and consistent:
                current,observations=stabilize_pose(current,prior,dt)
            # Keep slow size/angle stabilization, replace ONLY center motion.
            iris_filter=track_center(raw_iris[:2],prior.get('iris_filter') if consistent and not self.static else None,
                                     dt,float(np.mean(current[2:4])),motion_hint=pupil_hint)
            current[:2]=iris_filter['position']
            if reliable:
                normalized_support=normalize_support(envelope,center,rx,ry,geometry_angle)
                if not self.static and consistent:
                    normalized_support=stabilize_support(normalized_support,prior['support'],dt)
            else:
                normalized_support=prior['support']
            e.lens_opacity=opacity
            pupil_filter=None
            if pupil_params is None:
                # Do not trust a huge/off-center pupil mask. A previous credible
                # pupil can cover a brief failure; otherwise the material's hole remains.
                old_pupil=prior.get('pupil') if consistent else None
                old_pupil_time=prior.get('pupil_time',-1) if consistent else -1
                if old_pupil is not None and now-old_pupil_time<self.HOLD_SECONDS:
                    pupil_params=old_pupil
                    pupil_time=old_pupil_time
                    e.pupil_source='brief pupil fallback'
                else:pupil_time=-1
            else:
                pupil_time=now
                e.pupil_observed=raw_pupil
                e.pupil_source='pupil'
                pupil_filter=track_center(e.pupil_observed,prior.get('pupil_filter') if consistent and not self.static else None,
                                          dt,float(np.mean(current[2:4])),
                                          motion_hint=(raw_iris[:2]-prior['iris_filter']['raw']
                                                       if consistent and prior.get('iris_filter') is not None else None))
                # Store pupil offset relative to the independently filtered
                # iris only for bounded fallback; it does NOT deform the print.
                pupil_params[:2]=np.linalg.solve(pose_basis(current),pupil_filter['position']-current[:2])
                if not self.static and consistent and prior.get('pupil') is not None:
                    delta=pupil_params[2:]-prior['pupil'][2:]
                    change=float(np.max(np.abs(delta)))
                    if change<.015:pupil_params[2:]=prior['pupil'][2:]
                    else:
                        tau=.16 if change<.10 else .035
                        pupil_params[2:]=prior['pupil'][2:]+(1-np.exp(-dt/tau))*delta
            if pupil_params is not None:
                e.pupil_anchor=current[:2]+pose_basis(current)@pupil_params[:2]
            else:e.pupil_anchor=current[:2].copy()
            # One render center. Iris supplies axes/roll, accepted pupil supplies
            # translation. Keep the iris-only state separately for future checks.
            render_pose=current.copy()
            render_pose[:2]=e.pupil_anchor
            e.lens_pose=render_pose
            e.lens_support=support_at_pose(normalized_support,e.iris.shape,render_pose,e.affine)
            protection_params=None if pupil_params is None else pupil_params.copy()
            if protection_params is not None:protection_params[:2]=0.
            canonical_pupil=pupil_mask((128,128),np.array([64.,64.]),63.,63.,protection_params)
            e.pupil_protection=support_at_pose(canonical_pupil,e.iris.shape,render_pose,e.affine)
            self.history[e.side]={'pose':current,'time':now,'reliable_time':last_reliable,
                                 'pupil':pupil_params,'pupil_time':pupil_time,'support':normalized_support,
                                 'observations':observations,'iris_filter':iris_filter,'pupil_filter':pupil_filter,
                                 'flow_gray':cv2.cvtColor(e.crop,cv2.COLOR_BGR2GRAY),
                                 'flow_mask':((e.lens_support>.4)&(e.eyelid>0)).astype(np.uint8),
                                 'flow_center':e.center.copy()}
        for side in set(self.history)-seen:self.history.pop(side,None)


def iris_envelope(eye,center,rx,ry):
    """Close internal cracks without extrapolating a full disk onto skin/sclera."""
    yy,xx=np.indices(eye.iris.shape)
    ellipse=((xx-center[0])/rx)**2+((yy-center[1])/ry)**2
    probability=eye.iris_prob if eye.iris_prob is not None else eye.iris.astype(np.float32)
    evidence=((probability>=.30)&(ellipse<=1.12**2)&(eye.eyelid>0)).astype(np.uint8)
    n,labels,stats,_=cv2.connectedComponentsWithStats(evidence)
    valid=np.zeros_like(evidence)
    for label in range(1,n):
        if stats[label,cv2.CC_STAT_AREA]>=max(3,.01*np.pi*rx*ry):valid[labels==label]=1
    ys,xs=np.nonzero(valid)
    if len(xs)<12:return np.zeros_like(evidence,dtype=np.float32)
    hull=cv2.convexHull(np.column_stack([xs,ys]).astype(np.int32))
    filled=np.zeros_like(evidence)
    cv2.fillConvexPoly(filled,hull,1)
    # One narrow pixel-scale allowance at the inferred iris boundary.
    allowance=max(1,min(2,round(min(rx,ry)*.035)))
    filled=cv2.dilate(filled,np.ones((2*allowance+1,2*allowance+1),np.uint8))
    filled[(ellipse>1.05**2)|(eye.eyelid==0)]=0
    distance=cv2.distanceTransform(filled,cv2.DIST_L2,3)
    return np.clip(distance/1.0,0,1).astype(np.float32)


def normalize_support(mask,center,rx,ry,angle=0.):
    cs,sn=np.cos(angle),np.sin(angle)
    linear=np.array([[63/rx*cs,63/rx*sn],[-63/ry*sn,63/ry*cs]],np.float64)
    matrix=np.column_stack([linear,np.array([64.,64.])-linear@center])
    return cv2.warpAffine(mask,matrix,(128,128),flags=cv2.INTER_LINEAR)


def restore_support(mask,shape,center,rx,ry):
    matrix=np.array([[rx/63,0,center[0]-64*rx/63],[0,ry/63,center[1]-64*ry/63]],np.float64)
    return cv2.warpAffine(mask,matrix,(shape[1],shape[0]),flags=cv2.INTER_LINEAR)


def estimate_pupil(eye,center,rx,ry):
    """Accept a compact central pupil; reject iris/eyeglass misclassification."""
    yy,xx=np.indices(eye.iris.shape)
    r2=((xx-center[0])/rx)**2+((yy-center[1])/ry)**2
    # Use the pupil head directly, independent of a fragmented iris component.
    prob=eye.pupil_prob if eye.pupil_prob is not None else eye.pupil.astype(np.float32)
    raw=((prob>=.60)&(eye.eyelid>0)).astype(np.uint8)
    n,labels,stats,centroids=cv2.connectedComponentsWithStats(raw)
    candidates=[i for i in range(1,n) if stats[i,cv2.CC_STAT_AREA]>=8 and
                np.linalg.norm((centroids[i]-center)/[rx,ry])<.40]
    if not candidates:return None
    label=min(candidates,key=lambda i:np.linalg.norm((centroids[i]-center)/[rx,ry]))
    region=labels==label
    area_ratio=region.sum()/max(1,np.pi*rx*ry)
    if not .015<area_ratio<.52 or np.mean(r2[region]<.80**2)<.95:return None
    ys,xs=np.nonzero(region)
    cx,cy=float(xs.mean()),float(ys.mean())
    # External contour ignores internal specular holes. Reject a contour cut
    # heavily by eyelid boundaries rather than extrapolating an unseen pupil.
    contours,_=cv2.findContours(region.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contour=max(contours,key=cv2.contourArea)
    quality=.45
    if len(contour)>=8:
        points=contour[:,0,:].astype(np.float32)
        lid_distance=cv2.distanceTransform(eye.eyelid.astype(np.uint8),cv2.DIST_L2,3)
        clipped=np.mean(lid_distance[points[:,1].astype(int),points[:,0].astype(int)]<1.5)
        if clipped>.22:return None
        (fx,fy),(a,b),angle=cv2.fitEllipseDirect(contour)
        if min(a,b)<2 or not np.isfinite([fx,fy,a,b,angle]).all():return None
        phi=np.deg2rad(angle);cs,sn=np.cos(phi),np.sin(phi)
        shifted=points-[fx,fy]
        local=shifted@np.array([[cs,-sn],[sn,cs]])
        distances=np.sqrt((local[:,0]/(a/2))**2+(local[:,1]/(b/2))**2)
        residual=float(np.median(np.abs(distances-1)))
        bins=np.unique((np.mod(np.arctan2(local[:,1]/b,local[:,0]/a),2*np.pi)/(2*np.pi)*8).astype(int))
        if residual>.16 or len(bins)<6:return None
        if np.linalg.norm((np.array([fx,fy])-center)/[rx,ry])>.40:return None
        cx,cy=float(fx),float(fy)
        quality=float(np.clip(1-residual*3-clipped,.45,1))
    # A filled ellipse has sigma_x=a/2, sigma_y=b/2. A small subpixel
    # allowance protects its border without the previous whole-mask dilation.
    prx=max(1.,2*float(np.std(xs))+.6)
    pry=max(1.,2*float(np.std(ys))+.6)
    ratios=np.array([prx/rx,pry/ry])
    if np.max(ratios)>.76 or np.min(ratios)<.10 or max(ratios)/min(ratios)>1.9:return None
    eye.pupil_confidence=quality
    return np.array([(cx-center[0])/rx,(cy-center[1])/ry,*ratios],np.float64)


def pupil_mask(shape,center,rx,ry,params):
    if params is None:return np.zeros(shape,np.float32)
    yy,xx=np.indices(shape,dtype=np.float32)
    cx,cy=center+params[:2]*[rx,ry]
    ax,ay=np.maximum(params[2:]*[rx,ry],1)
    rr=np.sqrt(((xx-cx)/ax)**2+((yy-cy)/ay)**2)
    # Keep the pupil interior untouched; only a narrow OUTWARD transition.
    return (1-smoothstep(1.,1.+.5/max(1,min(ax,ay)),rr)).astype(np.float32)


def compose_lens(image,eye,material,strength=.70,scale=1.,frame_to_target=None):
    """Operate only in a bounded eye ROI. Masks always come from the same frame."""
    pose=getattr(eye,'lens_pose',None)
    if material is None or pose is None or not eye.is_open or strength<=0:return
    scale=clamp_lens_scale(scale)
    h,w=image.shape[:2]
    source_to_target=np.eye(3) if frame_to_target is None else frame_to_target
    inverse=cv2.invertAffineTransform(eye.affine)
    crop_to_target=(source_to_target @ np.vstack([inverse,[0,0,1]]))[:2]
    center=cv2.transform(pose[:2].astype(np.float32)[None,None],source_to_target[:2])[0,0]
    # Build texture-normalized -> target transform, including display scaling.
    angle=pose[4];cs,sn=np.cos(angle),np.sin(angle)
    basis=np.array([[cs*pose[2]*scale,-sn*pose[3]*scale],[sn*pose[2]*scale,cs*pose[3]*scale]])
    basis=source_to_target[:2,:2]@basis
    if abs(np.linalg.det(basis))<1e-6:return
    corners=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])@basis.T+center
    x0,y0=np.maximum(np.floor(corners.min(0)).astype(int)-2,[0,0])
    x1,y1=np.minimum(np.ceil(corners.max(0)).astype(int)+2,[w,h])
    if x1<=x0 or y1<=y0:return
    yy,xx=np.mgrid[y0:y1,x0:x1].astype(np.float32)
    coords=np.stack([xx-center[0],yy-center[1]],axis=-1)@np.linalg.inv(basis).T
    original_radius=np.linalg.norm(coords,axis=2)*scale
    hole=material.metadata.get('calibration',{}).get('hole',0.)
    hole=float(hole) if hole is not None else 0.
    hole=float(np.clip(hole,0,.8)) if np.isfinite(hole) else 0.
    # No inner/outer center offset or angular stretch: both boundaries share pose.
    texture_coords=concentric_uv(coords,hole,scale)
    size=material.rgba.shape[0]
    tex=cv2.remap(material.premultiplied,((texture_coords[:,:,0]+1)*size/2-.5).astype(np.float32),
                 ((texture_coords[:,:,1]+1)*size/2-.5).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    # Inward feather never creates color outside the eyelid.
    distance=cv2.distanceTransform(eye.eyelid.astype(np.uint8),cv2.DIST_L2,3)
    lid=np.clip(distance/1.5,0,1)
    pupil=getattr(eye,'pupil_protection',None)
    if pupil is None:
        # Normally populated by LensTracker. Avoid using unvalidated raw masks.
        pupil=np.zeros_like(lid)
    support=getattr(eye,'lens_support',None)
    if support is None:return
    masks=np.dstack([lid,1-pupil,support]).astype(np.float32)
    local=crop_to_target.copy();local[:,2]-=[x0,y0]
    warped=cv2.warpAffine(masks,local,(x1-x0,y1-y0),flags=cv2.INTER_LINEAR)
    footprint=warped[:,:,2]
    if scale>1.:
        # Expand the print's verified envelope about the unified render center.
        # Eyelid and pupil masks remain unscaled, so the print can reach visible
        # sclera but cannot grow onto the eyelid or shrink the real pupil.
        expanded=crop_to_target.copy()*scale
        expanded[:,2]+=(1-scale)*center
        expanded[:,2]-=[x0,y0]
        footprint=cv2.warpAffine(support,expanded,(x1-x0,y1-y0),flags=cv2.INTER_LINEAR)
    mask=np.clip(warped[:,:,0]*warped[:,:,1]*footprint,0,1)
    a=tex[:,:,3]*mask*float(np.clip(strength,0,1))*float(getattr(eye,'lens_opacity',1.))
    roi=image[y0:y1,x0:x1]
    rgb=roi[:,:,::-1].astype(np.float32)/255
    original=srgb_to_linear(rgb)
    lum=original@np.array([.2126,.7152,.0722],np.float32)
    # Do not let bright sclera newly covered by enlargement change the iris's
    # lighting reference, or the outer ring would wash out / pulse with size.
    valid=(warped[:,:,2]>.5)&(warped[:,:,1]>.8)&(a>.08)&(lum>.015)
    reference=float(np.median(lum[valid])) if valid.any() else .15
    lighting=np.clip((cv2.GaussianBlur(lum,(0,0),1.0)+.03)/(reference+.03),.65,1.35)
    if scale>1.:
        outside=smoothstep(.90,1.05,original_radius)
        lighting=lighting*(1-outside)+outside
    color=tex[:,:,:3]/np.maximum(tex[:,:,3:4],1e-6)
    color=np.clip(color*lighting[:,:,None],0,1)
    # Retain real corneal highlights. Near-white localized highlights receive
    # less material coverage; this is not a claim to classify every reflection.
    white=smoothstep(.62,.95,rgb.min(2))
    local_peak=smoothstep(.02,.12,lum-cv2.GaussianBlur(lum,(0,0),2.0))
    a*=1-.95*white*local_peak
    blended=original*(1-a[:,:,None])+color*a[:,:,None]
    result=np.rint(linear_to_srgb(blended)*255).astype(np.uint8)[:,:,::-1]
    changed=a>1e-6
    roi[changed]=result[changed]
