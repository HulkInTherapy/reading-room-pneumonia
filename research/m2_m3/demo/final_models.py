"""Load complete FP32 final checkpoints without downloading ImageNet weights."""
from pathlib import Path
import torch,psutil
from torch import nn
from torchvision import models
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/paper_online5/results'
MODEL_NAMES=['resnet18','densenet121','googlenet','inception_v3','alexnet']
def guard():
    if psutil.Process().memory_info().rss>1.4*1024**3:
        raise RuntimeError('Model process memory limit reached')
    if psutil.disk_usage(str(ROOT)).free<400*1024**2:
        raise RuntimeError('Disk space is low')
def load_model(name):
    kwargs={'weights':None}
    if name in ('googlenet','inception_v3'):
        kwargs.update(aux_logits=False,init_weights=False,transform_input=True)
    m=getattr(models,name)(**kwargs)
    if name=='resnet18':m.fc=nn.Linear(512,2)
    elif name=='densenet121':m.classifier=nn.Linear(1024,2)
    elif name=='alexnet':m.classifier=nn.Sequential(nn.Linear(9216,512),nn.ReLU(),nn.Dropout(.5),nn.Linear(512,2))
    elif name=='googlenet':m.fc=nn.Linear(1024,2)
    else:m.fc=nn.Linear(2048,2)
    m.load_state_dict(torch.load(OUT/f'{name}_final.pt',map_location='cpu',weights_only=True),strict=True)
    return m.eval()
