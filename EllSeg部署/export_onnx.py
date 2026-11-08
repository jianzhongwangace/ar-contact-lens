"""Export only the unchanged pretrained forward operations."""
import hashlib
import json
import torch
import onnx
from ellseg_core import ROOT,WEIGHTS,EllSegPredictor

def main():
    predictor=EllSegPredictor('cpu')
    output=ROOT/'weights/ellseg.onnx'
    with torch.inference_mode():
        torch.onnx.export(predictor.model,torch.zeros(1,1,240,320),str(output),
                          input_names=['eye'],output_names=['logits','regression'],
                          opset_version=17,do_constant_folding=True,dynamo=False)
    onnx.checker.check_model(str(output))
    metadata={'checkpoint_sha256':hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
              'onnx_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
              'input':[1,1,240,320],'precision':'float32','opset':17}
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
    print('Exported:',output)

if __name__=='__main__':main()
