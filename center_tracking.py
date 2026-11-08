"""Current-timestamp alpha-beta centers; no future-frame extrapolation/dead zone."""
import numpy as np

def track_center(observed,previous,dt,radius,motion_hint=None):
    observed=np.asarray(observed,np.float64)
    if previous is None or not 0<dt<.18:
        return {'position':observed.copy(),'velocity':np.zeros(2),'raw':observed.copy(),'pending':None}
    position=previous['position'];velocity=previous['velocity']
    prediction=position+velocity*dt
    innovation=observed-prediction
    displacement=observed-previous['raw']
    corroborated=False
    if motion_hint is not None:
        hint=np.asarray(motion_hint,np.float64)
        # Agreement between independent iris and pupil centers identifies a real
        # moving eye/head; do not treat the shared motion as an isolated outlier.
        corroborated=(np.linalg.norm(displacement)>max(.6,radius*.025)
                      and np.linalg.norm(hint)>max(.6,radius*.025)
                      and np.dot(displacement,hint)>0
                      and np.linalg.norm(displacement-hint)<max(1.5,radius*.12))
    if corroborated:
        alpha=1-np.exp(-dt/.010)
        result=prediction+alpha*innovation
        velocity=displacement/dt
        return {'position':result,'velocity':velocity,'raw':observed.copy(),'pending':None}
    jump=np.linalg.norm(innovation)>max(3.,radius*.24)
    pending=previous.get('pending')
    # A lone large outlier is rejected; a second agreeing observation confirms
    # a saccade. Keep current-time prediction for at most this one measurement.
    if jump:
        direction_ok=(pending is not None and
                      np.dot(observed-position,pending-position)>
                      .7*np.linalg.norm(observed-position)*np.linalg.norm(pending-position))
        if pending is None or not (np.linalg.norm(observed-pending)<max(1.5,radius*.14) or direction_ok):
            return {'position':prediction,'velocity':velocity*.5,'raw':observed.copy(),'pending':observed.copy()}
        velocity=(observed-pending)/dt
        speed=np.linalg.norm(velocity);maximum=max(30.,radius*20)
        if speed>maximum:velocity*=maximum/speed
        return {'position':observed.copy(),'velocity':velocity,'raw':observed.copy(),'pending':None}
    # A critically damped constant-velocity observer removes sustained-motion
    # lag without hard-locking small true displacements. Gains scale with dt.
    gain=1-np.exp(-dt/.18)
    alpha=2*gain-gain*gain
    beta=gain*gain
    result=prediction+alpha*innovation
    velocity=velocity+beta/max(dt,1e-6)*innovation
    maximum=max(30.,radius*20)
    speed=np.linalg.norm(velocity)
    if speed>maximum:velocity*=maximum/speed
    return {'position':result,'velocity':velocity,'raw':observed.copy(),'pending':None}

def pose_basis(pose):
    cs,sn=np.cos(pose[4]),np.sin(pose[4])
    return np.array([[cs*pose[2],-sn*pose[3]],[sn*pose[2],cs*pose[3]]],np.float64)

def concentric_uv(coords,hole,scale):
    """One center for both boundaries; radial enlargement preserves angular symmetry."""
    hole=float(np.clip(hole,0,.8))
    radius=np.linalg.norm(coords,axis=-1)
    original_radius=radius*max(scale,1.)
    mapped=np.where(original_radius<=hole,original_radius,
                    hole+(original_radius-hole)*(1-hole)/(max(scale,1.)-hole))
    return (coords*(mapped/np.maximum(radius,1e-8))[...,None]).astype(np.float32)


def annular_uv(coords,inner_offset,hole,scale):
    """Legacy comparison helper; NOT used by current rendering.

    Nested inner/outer-circle ray mapping. Outer rim is fixed, hole follows pupil.

    coords and offset are in the enlarged outer ellipse's unit coordinates.
    Each ray starts at the pupil center and ends on the fixed unit outer rim.
    A strict nested-circle constraint ensures positive radial derivative.
    """
    offset=np.asarray(inner_offset,np.float64)
    hole=float(np.clip(hole,0,.8))
    physical_hole=hole/max(scale,1.)
    if not np.isfinite(offset).all() or np.linalg.norm(offset)+physical_hole>=.96:
        return None
    vector=coords-offset
    radius=np.linalg.norm(vector,axis=-1)
    direction=vector/np.maximum(radius[...,None],1e-8)
    dot=direction@offset
    rim=-dot+np.sqrt(np.maximum(dot*dot+1-np.dot(offset,offset),1e-8))
    if physical_hole>1e-6:
        sampled=np.where(radius<=physical_hole,radius/physical_hole*hole,
                         hole+(radius-physical_hole)*(1-hole)/np.maximum(rim-physical_hole,1e-6))
    else:sampled=radius/np.maximum(rim,1e-6)
    return (direction*sampled[...,None]).astype(np.float32)
