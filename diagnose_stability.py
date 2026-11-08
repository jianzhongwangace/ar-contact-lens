"""Frozen-input comparison; no new user images or videos are saved."""
import argparse
import importlib.util
import json
from pathlib import Path
import cv2
import numpy as np
from eye_pipeline import EyePipeline,ROOT
from lens_import import LensImportControl
from lens_renderer import LensTracker,compose_lens,stabilize_pose
from video_app import VideoReader

class NoTracking:
    def update(self,*args,**kwargs):pass
    def reset(self):pass

def metrics(values):
    if len(values)<2:return None
    array=np.stack(values)
    return {'center_rms_std_px':float(np.linalg.norm(np.std(array[:,:2],axis=0))),
            'axes_mean_std_px':float(np.mean(np.std(array[:,2:4],axis=0))),
            'roll_std_degrees':float(np.rad2deg(np.std(np.unwrap(array[:,4]))))}

def synthetic_compare():
    truth=np.array([200.,150.,20.,18.,.1])
    state={'pose':truth.copy(),'observations':[truth.copy()]}
    previous=truth.copy();rng=np.random.default_rng(20260921)
    before=[];after=[]
    for i in range(200):
        raw=truth+rng.normal(0,[.65,.65,.55,.5,.025])
        old=previous+(1-np.exp(-(1/30)/.035))*(raw-previous)
        previous=old
        stable,observations=stabilize_pose(raw,state,1/30)
        state={'pose':stable,'observations':observations}
        if i>=20:before.append(old.copy());after.append(stable.copy())
    return {'scope':'controlled stationary landmark noise, not a live-camera measurement',
            'before':metrics(before),'after':metrics(after)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--frame',type=int,default=30)
    p.add_argument('--repeats',type=int,default=90)
    a=p.parse_args()
    reader=VideoReader(a.video)
    try:frame=reader.seek(a.frame)[0]
    finally:reader.close()
    spec=importlib.util.spec_from_file_location('before_stability',ROOT/'logs/lens_renderer_before_stabilization.py')
    before_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(before_module)
    old_tracker=before_module.LensTracker();new_tracker=LensTracker()
    pipeline=EyePipeline();pipeline.lens_tracker=NoTracking()
    control=LensImportControl()
    if control.material is None:raise RuntimeError('Select a material first')
    result={'scope':'same frozen real frame replayed through FaceMesh + LightIrisNet; not live motion accuracy',
            'frame_index':a.frame,'repeats':a.repeats,'eyes':{},'controlled_noise':synthetic_compare()}
    records={};fixed_crops={};previous_images={}
    try:
        for i in range(a.repeats):
            eyes,_=pipeline.process(frame,timestamp=i/30)
            old_tracker.update(eyes,i/30)
            old_poses={};old_images={}
            for e in eyes:
                if e.side not in fixed_crops:
                    affine=e.affine.copy()
                    crop=cv2.warpAffine(frame,affine,(e.crop.shape[1],e.crop.shape[0]))
                    fixed_crops[e.side]=(crop,np.vstack([affine,[0,0,1]]))
                    records[e.side]={'before':[],'after':[],'before_pixels':[],'after_pixels':[]}
                old_poses[e.side]=None if e.lens_pose is None else e.lens_pose.copy()
                crop,transform=fixed_crops[e.side]
                output=crop.copy()
                before_module.compose_lens(output,e,control.material,control.strength,control.lens_scale,transform)
                old_images[e.side]=output
            new_tracker.update(eyes,i/30)
            for e in eyes:
                crop,transform=fixed_crops[e.side]
                output=crop.copy()
                compose_lens(output,e,control.material,control.strength,control.lens_scale,transform)
                if i>=20:
                    if old_poses[e.side] is not None:records[e.side]['before'].append(old_poses[e.side])
                    if e.lens_pose is not None:records[e.side]['after'].append(e.lens_pose.copy())
                    if e.side in previous_images:
                        old_image,new_image=previous_images[e.side]
                        records[e.side]['before_pixels'].append(float(np.abs(old_images[e.side].astype(float)-old_image).mean()))
                        records[e.side]['after_pixels'].append(float(np.abs(output.astype(float)-new_image).mean()))
                previous_images[e.side]=(old_images[e.side],output)
        for side,r in records.items():
            result['eyes'][side]={'before':metrics(r['before']),'after':metrics(r['after']),
                'before_visible_samples':len(r['before']),'after_visible_samples':len(r['after']),
                'before_pixel_change_mean':float(np.mean(r['before_pixels'])) if r['before_pixels'] else None,
                'after_pixel_change_mean':float(np.mean(r['after_pixels'])) if r['after_pixels'] else None}
        (ROOT/'logs/stability-comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2))
    finally:pipeline.close()

if __name__=='__main__':main()
