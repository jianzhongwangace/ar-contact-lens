"""Same-observation center wiring comparison. This is NOT human-ground-truth accuracy."""
import argparse
import importlib.util
import json
import time
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from eye_pipeline import EyePipeline,ROOT,alignment_view
from lens_renderer import LensTracker,compose_lens
from lens_import import LensImportControl
from video_app import VideoReader
from center_tracking import track_center

class NoTracking:
    def reset(self):pass
    def update(self,*args,**kwargs):pass

def stats(items):
    return {'samples':len(items),'median':float(np.median(items)),'p95':float(np.percentile(items,95))} if items else None

def main():
    a=argparse.ArgumentParser();a.add_argument('--video',required=True);args=a.parse_args()
    spec=importlib.util.spec_from_file_location('prior_alignment',ROOT/'logs/lens_renderer_before_enlargement.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    old_tracker=old.LensTracker();new_tracker=LensTracker()
    pipeline=EyePipeline();pipeline.lens_tracker=NoTracking()
    reader=VideoReader(args.video);control=LensImportControl()
    samples={'old_error_px':[],'new_error_px':[],'old_error_iris_diameter_fraction':[],
             'new_error_iris_diameter_fraction':[],'processing_ms':[]}
    frames=0;observations=0;fallback=0;visible=0;selected=[]
    try:
        item=reader.first
        while item is not None:
            frame,stamp,index=item;start=time.perf_counter()
            eyes,_=pipeline.process(frame,timestamp=stamp)
            old_tracker.update(eyes,stamp)
            before={e.side:None if e.lens_pose is None else e.lens_pose.copy() for e in eyes}
            new_tracker.update(eyes,stamp)
            for eye in eyes:
                visible+=int(eye.lens_pose is not None)
                fallback+=int(eye.pupil_source=='brief pupil fallback')
                if eye.pupil_observed is None or eye.pupil_anchor is None:continue
                observations+=1
                if before[eye.side] is not None:
                    old_error=float(np.linalg.norm(before[eye.side][:2]-eye.pupil_observed))
                    new_error=float(np.linalg.norm(eye.pupil_anchor-eye.pupil_observed))
                    diameter=2*float(np.mean(eye.lens_pose[2:4]))
                    samples['old_error_px'].append(old_error);samples['new_error_px'].append(new_error)
                    samples['old_error_iris_diameter_fraction'].append(old_error/diameter)
                    samples['new_error_iris_diameter_fraction'].append(new_error/diameter)
                if index in (40,150,280) and control.material is not None:
                    result=eye.crop.copy()
                    compose_lens(result,eye,control.material,control.strength,control.lens_scale,np.vstack([eye.affine,[0,0,1]]))
                    selected.append((index,eye.side,eye.crop.copy(),alignment_view(eye),result))
            samples['processing_ms'].append((time.perf_counter()-start)*1000)
            frames+=1;item=reader.read()
        report={'scope':'Relative to the NEW accepted model pupil observation; not manual ground truth or guaranteed visual accuracy.',
                'frames':frames,'accepted_pupil_observations':observations,'brief_pupil_fallbacks':fallback,
                'visible_eye_samples':visible,'metrics':{key:stats(value) for key,value in samples.items()}}
        # Small persistent motions and constant-velocity tests without AI ambiguity.
        truth=np.array([100.,100.,20.,18.,0.]);prior={'pose':truth.copy(),'observations':[truth.copy()]};current=None
        old_lag=[];new_lag=[]
        for i in range(90):
            raw=truth.copy();raw[0]+=i/3
            filtered,history=old.stabilize_pose(raw,prior,1/30);prior={'pose':filtered,'observations':history}
            current=track_center(raw[:2],current,1/30,20)
            if i>30:old_lag.append(raw[0]-filtered[0]);new_lag.append(raw[0]-current['position'][0])
        report['synthetic_10px_per_second']={'before_median_error_px':float(np.median(old_lag)),
                                            'after_median_error_px':float(np.median(new_lag))}
        (ROOT/'logs/pupil-alignment-validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        if selected:
            sheet=Image.new('RGB',(1152,len(selected)*275+40),(24,27,32));draw=ImageDraw.Draw(sheet)
            font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
            for col,text in enumerate(['原眼','橙：模型瞳孔 / 绿：透明孔 / 青：虹膜','试戴']):
                draw.text((col*384+8,8),text,font=font,fill='white')
            for row,(index,side,*images) in enumerate(selected):
                for col,image in enumerate(images):
                    tile=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)).resize((384,240))
                    sheet.paste(tile,(col*384,40+row*275))
                draw.text((8,280+row*275),f'第 {index+1} 帧 · {side}',font=font,fill='white')
            sheet.save(ROOT/'logs/pupil-alignment-preview.png')
        print(json.dumps(report,indent=2))
    finally:reader.close();pipeline.close()

if __name__=='__main__':main()
