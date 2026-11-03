"""Check fresh image forwards against cached benchmark predictions, no tuning."""
import sys,json,gc
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from PIL import Image
from torchvision import transforms
from baseline import build_model,MODEL_NAMES,guard
rows=json.loads((HERE/'split.json').read_text())['test_records']
indices=[]
for label in [0,1]:
    candidates=[i for i,r in enumerate(rows) if r['label']==label]
    indices.extend([candidates[0],candidates[-1]])
tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
x=torch.stack([tf(Image.open(ROOT/rows[i]['path']).convert('RGB')) for i in indices])
checks=[]
for name in MODEL_NAMES:
    guard();prefix,tail,_=build_model(name)
    tail.load_state_dict(torch.load(HERE/'results'/f'{name}_best.pt',weights_only=True,map_location='cpu'));tail.eval()
    with torch.inference_mode():
        z=prefix(x).to(torch.float16).to(torch.float32);prefix.handle.remove()
        p=tail(z).softmax(-1)[:,1].numpy()
    expected=np.load(HERE/'results'/f'{name}_test_probabilities.npy')[indices]
    difference=float(np.max(np.abs(p-expected)))
    assert np.allclose(p,expected,atol=3e-5,rtol=1e-4),(name,difference)
    checks.append(dict(model=name,max_probability_difference=difference,scores=p.tolist()))
    del prefix,tail,z;gc.collect()
receipt=dict(verified=True,test_indices=indices,method='Fresh image normalization and full frozen-prefix/trainable-tail forwards; first/last examples of each class chosen without prediction-based selection',models=checks)
(HERE/'results/live_input_verification.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
