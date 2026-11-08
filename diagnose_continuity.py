"""Compare old/new composition on the SAME decoded frames and AI outputs."""
import importlib.util
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from eye_pipeline import EyePipeline,ROOT
from lens_import import LensImportControl
from lens_renderer import LensTracker,compose_lens
from video_app import VideoReader

class NoTracking:
    def update(self,*args,**kwargs):pass
    def reset(self):pass

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--video',type=Path,required=True)
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('old_lens',ROOT/'logs/lens_renderer_before_continuity_fix.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    before=old.LensTracker();after=LensTracker()
    pipeline=EyePipeline();pipeline.lens_tracker=NoTracking()
    reader=VideoReader(args.video)
    control=LensImportControl()
    if control.material is None:raise RuntimeError('Choose a material first')
    stats={'frames':0,'eye_samples':0,'before_visible':0,'after_visible':0,'bounded_fallback':0,
           'before_switches':0,'after_switches':0,'newly_rejected':0}
    last={};selected=[]
    try:
        item=reader.first
        while item is not None:
            frame,stamp,index=item
            eyes,_=pipeline.process(frame,timestamp=stamp)
            before.update(eyes,stamp)
            old_states={};old_images={}
            for e in eyes:
                old_states[e.side]=e.lens_pose is not None
                image=e.crop.copy()
                old.compose_lens(image,e,control.material,control.strength,control.lens_scale,
                                 np.vstack([e.affine,[0,0,1]]))
                old_images[e.side]=image
            after.update(eyes,stamp)
            for e in eyes:
                was=old_states[e.side];now=e.lens_pose is not None
                stats['eye_samples']+=1;stats['before_visible']+=int(was);stats['after_visible']+=int(now)
                stats['bounded_fallback']+=int(e.tracking_state=='brief fallback')
                stats['newly_rejected']+=int(was and not now)
                if e.side in last:
                    stats['before_switches']+=int(was!=last[e.side][0]);stats['after_switches']+=int(now!=last[e.side][1])
                last[e.side]=(was,now)
                current=e.crop.copy()
                compose_lens(current,e,control.material,control.strength,control.lens_scale,
                             np.vstack([e.affine,[0,0,1]]))
                old_pixels=np.any(old_images[e.side]!=e.crop,axis=2)
                new_pixels=np.any(current!=e.crop,axis=2)
                added=int(np.sum(new_pixels&~old_pixels))
                if was and now and added>10:
                    selected.append((added,index,e.side,e.crop.copy(),old_images[e.side],current))
                    selected=sorted(selected,key=lambda x:x[0],reverse=True)[:12]
            stats['frames']+=1
            item=reader.read()
        stats['scope']='Counts compare visibility/continuity, not ground-truth segmentation or physical lens accuracy.'
        stats['material']=str(control.path)
        (ROOT/'logs/continuity-comparison.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
        chosen=[]
        for entry in selected:
            if all(abs(entry[1]-x[1])>8 for x in chosen):chosen.append(entry)
            if len(chosen)>=3:break
        if chosen:
            sheet=Image.new('RGB',(1152,len(chosen)*285+45),(26,28,33))
            draw=ImageDraw.Draw(sheet);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
            for i,name in enumerate(['原眼','修正前','修正后']):draw.text((i*384+12,8),name,font=font,fill='white')
            for row,entry in enumerate(chosen):
                for col,img in enumerate(entry[3:]):
                    tile=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB));tile=tile.resize((384,240))
                    sheet.paste(tile,(384*col,row*285+45))
                draw.text((12,row*285+289),f'视频第 {entry[1]+1} 帧 · {entry[2]} eye',font=font,fill='white')
            sheet.save(ROOT/'logs/continuity-before-after.png')
        print(json.dumps(stats,ensure_ascii=False,indent=2))
    finally:reader.close();pipeline.close()

if __name__=='__main__':main()
