"""Create a minimal, attributed inference module from the pinned upstream source."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
REPO = ROOT / 'research' / 'LightIrisNet'
source = (REPO / 'src/irisseg/_test_impl.py').read_text(encoding='utf-8')
classes = source[source.index('class ResNet50Encoder'):source.index('# Inference utils')]
classes = classes.replace('weights=ResNet50_Weights.IMAGENET1K_V2', 'weights=None')
classes = classes.replace('weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1', 'weights=None')
classes = classes.replace('ResNet50Encoder(ResNet50_Weights.IMAGENET1K_V2)', 'ResNet50Encoder(None)')
target = ROOT / 'vendor'
target.mkdir(exist_ok=True)
header = '''"""LightIrisNet model definitions, MIT licensed.
Source: https://github.com/naveengv7/LightIrisNet
Commit: 034c807796a881173519c9984cae341dbb2e632d
Adaptation: only model classes extracted; ImageNet download disabled because
the full supplied checkpoint is loaded strictly. No retraining or changed layers.
See LIGHTIRISNET_LICENSE.txt.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights

'''
(target / 'lightiris_model.py').write_text(header + classes, encoding='utf-8')
(target / 'LIGHTIRISNET_LICENSE.txt').write_text((REPO / 'LICENSE').read_text(), encoding='utf-8')
checkpoint = REPO / 'model/best_mobilenetv3.pt'
manifest = {
    'repository': 'https://github.com/naveengv7/LightIrisNet',
    'commit': '034c807796a881173519c9984cae341dbb2e632d',
    'checkpoint': str(checkpoint.relative_to(ROOT)),
    'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    'upstream_test_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
    'adaptation': 'Extract model definitions; disable redundant ImageNet initialization downloads.'
}
(target / 'model_source.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print('Prepared model classes and license; checkpoint SHA256:', manifest['sha256'])
