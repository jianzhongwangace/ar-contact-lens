"""Same-inference head-turn regression; counts are continuity, not accuracy."""
import copy
import gzip
import importlib.util
import json
import pickle
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from eye_pipeline import EyePipeline, ROOT
from video_app import VideoReader
from lens_renderer import LensTracker, estimate_pupil
from ellseg_adapter import validate_geometry


def load_old(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    cache=Path('D:/ARLensRuntime/head-turn-observations.pkl.gz')
    if not cache.exists():
        pipeline=EyePipeline(device='auto',geometry='ellseg')
        pipeline.lens_tracker=SimpleNamespace(update=lambda *a,**k:None,reset=lambda:None)
        reader=VideoReader(next(ROOT.glob('微信视频*.mp4')))
        observations=[]
        try:
            item=reader.first
            while item is not None:
                frame,stamp,index=item
                eyes,_=pipeline.process(frame,timestamp=stamp)
                observations.append((stamp,copy.deepcopy(eyes)))
                if index%80==0:print('inference',index,flush=True)
                item=reader.read()
        finally:reader.close();pipeline.close()
        with gzip.open(cache,'wb',compresslevel=1) as f:pickle.dump(observations,f)
    else:
        with gzip.open(cache,'rb') as f:observations=pickle.load(f)
    old=load_old('old_lens',ROOT/'logs/lens_renderer_before_turn_fix.py').LensTracker()
    old_gate=load_old('old_gate',ROOT/'logs/ellseg_adapter_before_turn_fix.py').validate_geometry
    new=LensTracker();counts=[Counter(),Counter()];switches=[0,0];states=[{},{}];details=[]
    for index,(stamp,eyes) in enumerate(observations):
        for version,(tracker,gate) in enumerate(((old,old_gate),(new,validate_geometry))):
            current=copy.deepcopy(eyes)
            for e in current:
                e.geometry_ellipse,e.geometry_reason=gate(e,e.ellseg_result)
            tracker.update(current,stamp)
            for e in current:
                counts[version][e.tracking_state]+=1
                visible=e.lens_pose is not None and e.lens_opacity>.01
                if e.side in states[version] and visible!=states[version][e.side]:switches[version]+=1
                states[version][e.side]=visible
                if version==1 and (not visible or e.tracking_state!='tracked'):
                    r=e.ellseg_result;ell=r['ellipses']['iris']['ellipse']
                    pupil=None if ell is None else estimate_pupil(e,np.array(ell[:2]),ell[2],ell[3])
                    details.append({'frame':index,'eye':e.side,'state':e.tracking_state,
                                    'reason':e.tracking_reason,'geometry':e.geometry_reason,
                                    'light_pixels':int(e.iris.sum()),'ellseg_pixels':int((r['mask']>0).sum()),
                                    'pupil_valid':pupil is not None})
    report={'frames':len(observations),'before':dict(counts[0]),'after':dict(counts[1]),
            'visibility_switches_before_after':switches,'scope':'same model outputs, no annotated ground truth',
            'remaining':details}
    (ROOT/'logs/head-turn-comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='remaining'},indent=2),flush=True)
    print(Counter((d['geometry'],d['pupil_valid']) for d in details),flush=True)


if __name__=='__main__':main()
