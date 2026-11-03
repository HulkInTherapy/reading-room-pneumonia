"""Local research demo. All outputs come from saved learned models."""
import base64,gc,io,json,sys,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from PIL import Image,UnidentifiedImageError
from torchvision import transforms
from fastapi import FastAPI,UploadFile,File,Form,HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from demo.final_models import load_model,guard,MODEL_NAMES,OUT
torch.set_num_threads(2)
app=FastAPI();lock=threading.Lock();HERE=Path(__file__).resolve().parent
app.mount('/assets',StaticFiles(directory=HERE/'assets'),name='assets')
manifest=json.loads((ROOT/'data/manifest.json').read_text());test=[r for r in manifest if r['split']=='test']
# Fixed representative indices, not chosen according to prediction correctness.
samples=[]
for c in [0,1]:
    candidates=[(i,r) for i,r in enumerate(test) if r['label']==c]
    for j in [0,len(candidates)//2,len(candidates)-1]:
        i,r=candidates[j];samples.append({'id':str(i),'name':f'Test case {i+1}','truth':r['label'],'path':r['path']})
def ready():return (OUT/'summary.json').exists() and all((OUT/f'{n}_final.pt').exists() for n in MODEL_NAMES)
def png(im):
    b=io.BytesIO();im.save(b,format='PNG');return 'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()
@app.get('/')
def index():return FileResponse(HERE/'index.html')
@app.get('/api/status')
def status():return dict(ready=ready(),completed_models=[n for n in MODEL_NAMES if (OUT/f'{n}_final.pt').exists()],model='Five CNNs, majority vote',heatmap='ResNet18 component Grad-CAM')
@app.get('/api/samples')
def list_samples():return [dict(id=r['id'],name=r['name'],image=f"/api/sample/{r['id']}") for r in samples]
@app.get('/api/sample/{sample_id}')
def sample_image(sample_id:str):
    s=next((r for r in samples if r['id']==sample_id),None)
    if s is None:raise HTTPException(404,'Unknown sample')
    return FileResponse(ROOT/s['path'])
def input_tensor(im):
    return transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])(im).unsqueeze(0)
def gradcam_resnet(im,label):
    model=load_model('resnet18')
    activation=[];gradient=[]
    def hook(module,inputs,out):
        activation.append(out)
        out.register_hook(lambda g:gradient.append(g))
    handle=model.layer4[-1].register_forward_hook(hook)
    x=input_tensor(im).requires_grad_(True);model.zero_grad(set_to_none=True)
    logits=model(x);logits[0,label].backward()
    a=activation[0].detach();g=gradient[0].detach();cam=torch.relu((g.mean((2,3),keepdim=True)*a).sum(1,keepdim=True))
    cam=nn_interpolate(cam);cam=cam[0,0].numpy();maximum=float(cam.max());cam=cam/maximum if maximum>0 else np.zeros_like(cam)
    handle.remove();del model,x,logits,a,g;gc.collect();guard()
    rgb=np.array(im,dtype=float)/255
    # A conventional gradient color map; brightness derives only from real CAM.
    import matplotlib
    colors=matplotlib.colormaps['turbo'](cam)[:,:,:3]
    alpha=.45*cam[...,None]
    overlay=(rgb*(1-alpha)+colors*alpha)*255
    left=float(cam[:,:112].sum());right=float(cam[:,112:].sum());total=left+right
    return png(Image.fromarray(overlay.astype(np.uint8))),dict(method='Grad-CAM',component='ResNet18',target_class='Pneumonia' if label else 'Normal',source_resolution='7 x 7, upscaled to 224 x 224',image_left_activation_share=(left/total if total else None),image_right_activation_share=(right/total if total else None),positive_activation=maximum>0)
def nn_interpolate(x):return torch.nn.functional.interpolate(x,size=(224,224),mode='bilinear',align_corners=False)
def infer(im):
    started=time.time();x=input_tensor(im);probs={}
    for name in MODEL_NAMES:
        guard();model=load_model(name)
        with torch.inference_mode():
            probs[name]=float(model(x).softmax(-1)[0,1])
        del model;gc.collect()
    votes=sum(p>=.5 for p in probs.values());label=int(votes>=3);mean=float(np.mean(list(probs.values())))
    heat,explanation=gradcam_resnet(im,label)
    explanation.update(component_pneumonia_score=probs['resnet18'],component_vote='Pneumonia' if probs['resnet18']>=.5 else 'Normal')
    return dict(prediction='Pneumonia' if label else 'Normal',predicted_label=label,pneumonia_votes=votes,total_models=5,pneumonia_score=mean,model_probabilities=probs,heatmap=heat,explanation=explanation,input_image=png(im),seconds=time.time()-started,note='Research model prediction. Score is uncalibrated. Heatmap explains one component, not a lesion or the entire ensemble.')
@app.post('/api/predict')
async def predict(file:UploadFile|None=File(None),sample_id:str|None=Form(None)):
    if not ready():raise HTTPException(503,'Final baseline checkpoints are unavailable on this server.')
    if not lock.acquire(blocking=False):raise HTTPException(409,'One image is already processing. Please wait.')
    try:
        truth=None
        if file is not None:
            raw=await file.read(12*1024**2+1)
            if len(raw)>12*1024**2:raise HTTPException(413,'Image must be smaller than 12 MB')
            try:
                im=Image.open(io.BytesIO(raw))
                if im.format not in ['JPEG','PNG']:raise HTTPException(415,'Use a JPEG or PNG X-ray image')
                if im.width*im.height>25_000_000:raise HTTPException(413,'Image resolution is too large')
                im=im.convert('RGB').resize((224,224),Image.Resampling.BILINEAR)
            except (UnidentifiedImageError,OSError):raise HTTPException(400,'This file is not a readable JPEG or PNG image')
        else:
            s=next((r for r in samples if r['id']==sample_id),None)
            if s is None:raise HTTPException(400,'Choose a sample or upload an image')
            im=Image.open(ROOT/s['path']).convert('RGB');truth=s['truth']
        result=infer(im)
        if truth is not None:result['known_test_label']='Pneumonia' if truth else 'Normal';result['matches_known_label']=result['predicted_label']==truth
        return result
    finally:lock.release()

@app.get('/api/results')
def results():
    baseline=json.loads((OUT/'summary.json').read_text())
    path=ROOT.parent/'m4_m5/results/summary.json'
    vpath=ROOT.parent/'m4_m5/results/verification.json'
    return {'baseline':baseline,'experiments':json.loads(path.read_text()) if path.exists() else None,'verification':json.loads(vpath.read_text()) if vpath.exists() else None}

@app.post('/api/experiment/predict')
async def experiment_predict(file:UploadFile|None=File(None),sample_id:str|None=Form(None)):
    from demo import pilot
    if not pilot.ready():raise HTTPException(503,'The measured lung-crop experiment is not complete yet.')
    if not lock.acquire(blocking=False):raise HTTPException(409,'An image is already processing. Please wait.')
    try:
        started=time.time();truth=None
        if file is not None:
            raw=await file.read(12*1024**2+1)
            if len(raw)>12*1024**2:raise HTTPException(413,'Image must be smaller than 12 MB')
            try:
                im=Image.open(io.BytesIO(raw))
                if im.format not in ['JPEG','PNG']:raise HTTPException(415,'Use JPEG or PNG')
                if im.width*im.height>25_000_000:raise HTTPException(413,'Image resolution is too large')
                full,crop,mask,meta=pilot.segment(im)
            except (UnidentifiedImageError,OSError):raise HTTPException(400,'Unreadable image')
        else:
            s=next((r for r in samples if r['id']==sample_id),None)
            if s is None:raise HTTPException(400,'Choose a sample or upload an image')
            sha=Path(s['path']).stem;truth=s['truth']
            full=Image.open(pilot.DATA/'full'/f'{sha}.png').convert('RGB')
            crop=Image.open(pilot.DATA/'crop'/f'{sha}.png').convert('RGB')
            mask=Image.open(pilot.DATA/'masks'/f'{sha}.png').convert('L')
            meta=json.loads((pilot.DATA/'crop_metadata.json').read_text())[sha]
        preprocessing_seconds=time.time()-started;classification_started=time.time()
        conditions,components=pilot.classify(full,crop)
        classifier_seconds=time.time()-classification_started
        return dict(conditions=conditions,components=components,full_image=png(full),crop_image=png(crop),mask_image=png(mask),crop_metadata=meta,known_test_label=truth,seconds=time.time()-started,preprocessing_seconds=preprocessing_seconds,classifier_seconds=classifier_seconds,model='M4 frozen pretrained encoders with five-epoch heads',note='Sample crops are preprocessed from original source pixels; classifier inference runs now. Scores are uncalibrated.')
    finally:lock.release()
