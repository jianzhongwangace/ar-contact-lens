"""Check DirectML FP32 outputs against original PyTorch on identical inputs."""
import json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from eye_pipeline import EyePipeline, LightIrisPredictor, DEFAULT_WEIGHTS, ROOT
from onnx_predictor import OnnxIrisPredictor

def main():
    image=cv2.imdecode(np.fromfile(str(ROOT/'test_assets/astronaut.png'),np.uint8),1)
    pipeline=EyePipeline(device='cpu',static_image=True)
    try:eyes,_=pipeline.process(image)
    finally:pipeline.close()
    cpu=pipeline.predictor
    dml=OnnxIrisPredictor(DEFAULT_WEIGHTS,profile=True)
    base=[e.crop for e in eyes]
    assert len(base)==2
    cases={'original':base,
           'darker':[np.clip(c.astype(float)*.65,0,255).astype(np.uint8) for c in base],
           'brighter':[np.clip(c.astype(float)*1.25,0,255).astype(np.uint8) for c in base],
           'flipped':[cv2.flip(c,1) for c in base],
           'single_eye':[base[0]],
           'blank':[np.zeros_like(base[0]),np.zeros_like(base[1])]}
    results={}
    for name,crops in cases.items():
        a=cpu.predict(crops);b=dml.predict(crops)
        results[name]=[]
        for label,x,y in zip(['iris','pupil'],a,b):
            delta=np.abs(x-y)
            mismatch=float(np.mean((x>=.5)!=(y>=.5)))
            results[name].append({'head':label,'max_abs_error':float(delta.max()),
                                  'mean_abs_error':float(delta.mean()),'mask_disagreement_fraction':mismatch})
            assert np.isfinite(y).all()
            assert delta.max()<.005,(name,label,delta.max())
            assert mismatch<.001,(name,label,mismatch)
    profile=Path(dml.session.end_profiling())
    events=json.loads(profile.read_text())
    providers=Counter(e.get('args',{}).get('provider') for e in events if e.get('cat')=='Node' and e.get('args',{}).get('provider'))
    assert providers['DmlExecutionProvider']>0,'No nodes executed on DirectML'
    report={'adapter':dml.adapter,'precision':'FP32; no quantization','cases':results,
            'profile_provider_events':dict(providers),'profile':str(profile),
            'result':'passed','scope':'numerical equivalence; not segmentation accuracy'}
    (ROOT/'logs/acceleration-validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
