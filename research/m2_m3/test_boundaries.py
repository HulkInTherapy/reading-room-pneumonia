"""Check cached-prefix/tail composition against full torchvision forwards."""
import gc,json,torch
from pathlib import Path
from baseline import build_model,MODEL_NAMES,ROOT,guard
out=[]
for name in MODEL_NAMES:
    prefix,tail,boundary=build_model(name);tail.eval();x=torch.randn(2,3,224,224)
    with torch.inference_mode():z=prefix(x)
    prefix.handle.remove();full=prefix.model
    if name=='densenet121':full.classifier=tail.fc
    elif name=='alexnet':full.classifier=tail
    elif name=='resnet18':full.fc=tail[3]
    else:full.fc=tail[4]
    full.eval()
    with torch.inference_mode():a=tail(z);b=full(x)
    assert a.shape==(2,2) and torch.allclose(a,b,atol=1e-5,rtol=1e-5),(name,(a-b).abs().max())
    n=sum(p.numel() for p in tail.parameters() if p.requires_grad)
    out.append(dict(model=name,prefix_shape=list(z.shape),trainable_boundary=boundary,trainable_parameters=n,max_forward_difference=float((a-b).abs().max())))
    print('PASS',out[-1],flush=True);del prefix,tail,full,x,z,a,b;gc.collect();guard()
(ROOT/'results/architecture_verification.json').write_text(json.dumps(out,indent=2))
