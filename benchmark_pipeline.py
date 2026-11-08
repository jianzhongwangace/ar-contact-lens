"""Repeatable public-fixture benchmark; no camera images saved."""
import argparse
import importlib.util
import json
import time
from pathlib import Path
import cv2
import numpy as np

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--module',default='eye_pipeline.py')
    p.add_argument('--device',default='cpu')
    p.add_argument('--report',required=True)
    p.add_argument('--samples',type=int,default=20)
    a=p.parse_args()
    import sys
    spec=importlib.util.spec_from_file_location('bench_pipeline_module',a.module)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    module.ROOT=Path(__file__).resolve().parent
    image=cv2.imdecode(np.fromfile('test_assets/astronaut.png',np.uint8),1)
    # Scale public photo into 1080p canvas, ensuring the same pixels in every run.
    image=cv2.resize(image,(1080,1080))
    frame=np.zeros((1080,1920,3),np.uint8)
    frame[:,420:1500]=image
    pipeline=module.EyePipeline(weights=module.ROOT/'research/LightIrisNet/model/best_mobilenetv3.pt',device=a.device)
    results=[]
    try:
        for i in range(a.samples+3):
            start=time.perf_counter()
            eyes,t=pipeline.process(frame)
            midpoint=time.perf_counter()
            out=module.render(frame,eyes,t,0)
            end=time.perf_counter()
            if i>=3:
                results.append({'pipeline_ms':(midpoint-start)*1000,'render_ms':(end-midpoint)*1000,
                                'total_ms':(end-start)*1000,'mesh_ms':t['mesh_ms'],'iris_ms':t['iris_ms']})
        report={'device':a.device,'module':a.module,'samples':len(results),'capture_shape':[1920,1080],
                'eye_input':[256,160],'fixture':'public skimage astronaut; not camera FPS',
                'median_ms':{k:float(np.median([r[k] for r in results])) for k in results[0]},
                'p95_ms':{k:float(np.percentile([r[k] for r in results],95)) for k in results[0]},
                'eyes':len(eyes),'iris_pixels':[int(e.iris.sum()) for e in eyes]}
        Path(a.report).write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
    finally:pipeline.close()

if __name__=='__main__':main()
