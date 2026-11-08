"""Export the exact FP32 LightIrisNet main branches; no retraining or quantization."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import onnx
from eye_pipeline import LightIrisPredictor, ROOT

class SegmentationExport(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model=model

    def forward(self,x):
        m=self.model
        f=m.enc(x)
        z=m.dec(m.aspp(f['s32']),f['s4'])
        z=torch.nn.functional.interpolate(z,size=x.shape[-2:],mode='bilinear',align_corners=False)
        return m.head_iris(z).sigmoid(),m.head_pupil(m.pre_pupil(z)).sigmoid()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--width',type=int,default=256)
    p.add_argument('--height',type=int,default=160)
    a=p.parse_args()
    model=LightIrisPredictor(device='cpu')
    wrapper=SegmentationExport(model.model).eval()
    out=ROOT/'models'/f'lightiris-2x{a.height}x{a.width}.onnx'
    out.parent.mkdir(exist_ok=True)
    dummy=torch.zeros(2,3,a.height,a.width)
    with torch.inference_mode():
        torch.onnx.export(wrapper,dummy,str(out),input_names=['eyes'],
                          output_names=['iris','pupil'],opset_version=17,
                          do_constant_folding=True,dynamo=False)
    onnx.checker.check_model(str(out))
    manifest={'format':'FP32 ONNX opset 17','batch':2,'height':a.height,'width':a.width,
              'source_checkpoint_sha256':hashlib.sha256((ROOT/'research/LightIrisNet/model/best_mobilenetv3.pt').read_bytes()).hexdigest(),
              'onnx_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
    out.with_suffix('.json').write_text(json.dumps(manifest,indent=2))
    print('Exported and checked:',out,flush=True)

if __name__=='__main__':main()
