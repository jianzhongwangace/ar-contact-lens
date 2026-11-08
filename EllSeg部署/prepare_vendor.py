"""Reproducibly extract MIT upstream inference classes; keep training imports out."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'research/EllSeg'
COMMIT='8f1ea13336fa9c662403bd5823ac13cc6a6dd632'

def extract(path,names):
    source=path.read_text(encoding='utf-8')
    tree=ast.parse(source)
    chunks=[ast.get_source_segment(source,node) for node in tree.body
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
    if len(chunks)!=len(names):raise ValueError('Upstream structure changed')
    return '\n\n'.join(chunks).replace('np.int(', 'int(')

def main():
    (ROOT/'vendor').mkdir(exist_ok=True)
    (ROOT/'weights').mkdir(exist_ok=True)
    (ROOT/'upstream_reference').mkdir(exist_ok=True)
    model=SOURCE/'models/RITnet_v3.py'
    utils=SOURCE/'utils.py'
    header='''"""EllSeg inference blocks derived from RSKothari/EllSeg, MIT license.
Commit: 8f1ea13336fa9c662403bd5823ac13cc6a6dd632
Changes: np.int -> int; remove training/loss imports; inference-only wrapper.
Weights and encoder/decoder/regression forward operations are unchanged.
See License.md and source.json.
"""
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

'''
    blocks=extract(utils,{'regressionModule','convBlock'})+'\n\n'+extract(model,{
        'getSizes','Transition_down','DenseNet2D_down_block','DenseNet2D_up_block',
        'DenseNet_encoder','DenseNet_decoder'})
    wrapper='''

class EllSegNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.sizes=getSizes(32,1.2)
        self.enc=DenseNet_encoder(in_c=1,chz=32,actfunc=F.leaky_relu,growth=1.2,norm=nn.InstanceNorm2d)
        self.dec=DenseNet_decoder(chz=32,out_c=3,growth=1.2,actfunc=F.leaky_relu,norm=nn.InstanceNorm2d)
        self.elReg=regressionModule(self.sizes)

    def forward(self,frame):
        x4,x3,x2,x1,x=self.enc(frame)
        regression=self.elReg(x,0)
        logits=self.dec(x4,x3,x2,x1,x)
        return logits,regression
'''
    (ROOT/'vendor/ellseg_model.py').write_text(header+blocks+wrapper,encoding='utf-8')
    shutil.copy2(SOURCE/'License.md',ROOT/'vendor/License.md')
    for relative in ['models/RITnet_v3.py','utils.py','loss.py','evaluate_ellseg.py','README.md']:
        shutil.copy2(SOURCE/relative,ROOT/'upstream_reference'/Path(relative).name)
    shutil.copy2(SOURCE/'weights/all.git_ok',ROOT/'weights/all.git_ok')
    manifest={'repository':'https://github.com/RSKothari/EllSeg','commit':COMMIT,
        'checkpoint':'weights/all.git_ok','checkpoint_sha256':hashlib.sha256((ROOT/'weights/all.git_ok').read_bytes()).hexdigest(),
        'source_files':{str(p.relative_to(SOURCE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [model,utils]},
        'changes':['Extract inference blocks','np.int replaced by int','eval() used for inference',
                   'No training loss or data dependencies; no retraining'],
        'ellipse_mapping':'Follow evaluate_ellseg.py: iris regression[2:5], pupil regression[7:10]; comments in upstream utils conflict with runtime ordering.'}
    (ROOT/'vendor/source.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
