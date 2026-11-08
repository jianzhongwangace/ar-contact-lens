"""FP32 ONNX inference with explicit DirectML hardware selection."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
from gpu_info import preferred_adapter

ROOT=Path(__file__).resolve().parent

class OnnxIrisPredictor:
    def __init__(self,weights,width=256,height=160,device='dml',threads=4,profile=False):
        path=ROOT/'models'/f'lightiris-2x{height}x{width}.onnx'
        if not path.is_file():
            raise FileNotFoundError(f'{path.name} missing; run export_onnx.py --width {width} --height {height}')
        manifest=json.loads(path.with_suffix('.json').read_text())
        for file,key in [(path,'onnx_sha256'),(Path(weights),'source_checkpoint_sha256')]:
            if hashlib.sha256(file.read_bytes()).hexdigest()!=manifest[key]:
                raise RuntimeError(f'ONNX/source checksum mismatch: {file.name}')
        options=ort.SessionOptions()
        options.enable_mem_pattern=False
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads=threads
        options.enable_profiling=profile
        options.profile_file_prefix=str(ROOT/'logs'/'onnx-profile')
        self.device=device
        self.adapter=None
        if device=='dml':
            if 'DmlExecutionProvider' not in ort.get_available_providers():
                raise RuntimeError('Install onnxruntime-directml in the project environment')
            self.adapter=preferred_adapter()
            providers=[('DmlExecutionProvider',{'device_id':str(self.adapter['index'])}),'CPUExecutionProvider']
        else:providers=['CPUExecutionProvider']
        self.session=ort.InferenceSession(str(path),sess_options=options,providers=providers)
        self.session.disable_fallback()
        if device=='dml' and self.session.get_providers()[0]!='DmlExecutionProvider':
            raise RuntimeError('DirectML initialization failed; use --device cpu explicitly')
        self.parameters=4_229_858 # Complete source checkpoint, including unused auxiliary heads.
        self.shape=(height,width)
        self.mean=np.array([.485,.456,.406],dtype=np.float32).reshape(1,3,1,1)
        self.std=np.array([.229,.224,.225],dtype=np.float32).reshape(1,3,1,1)

    def predict(self,crops):
        n=len(crops)
        if n not in (1,2):raise ValueError('ONNX runtime accepts one or two eye crops')
        if any(c.shape[:2]!=self.shape for c in crops):raise ValueError('Eye crop shape differs from exported model')
        # Fixed batch=2 avoids dynamic allocation/compilation. Slice off duplicated eye.
        padded=crops if n==2 else [crops[0],crops[0]]
        rgb=np.stack([cv2.cvtColor(c,cv2.COLOR_BGR2RGB) for c in padded])
        x=np.ascontiguousarray(rgb.transpose(0,3,1,2),dtype=np.float32)/np.float32(255)
        x=np.ascontiguousarray((x-self.mean)/self.std)
        iris,pupil=self.session.run(['iris','pupil'],{'eyes':x})
        return iris[:n,0],pupil[:n,0]
