"""Real API inference, upload equivalence, score integrity and image validation."""
import base64,io,json
from pathlib import Path
import numpy as np,httpx
from PIL import Image
ROOT=Path(__file__).resolve().parent;URL='http://127.0.0.1:8765'
with httpx.Client(timeout=120) as c:
    assert c.get(URL+'/api/status').json()['ready']
    samples=c.get(URL+'/api/samples').json();assert len(samples)==6
    chosen=samples[3];index=int(chosen['id'])
    r=c.post(URL+'/api/predict',data={'sample_id':chosen['id']});r.raise_for_status();d=r.json()
    names=['resnet18','densenet121','googlenet','inception_v3','alexnet']
    expected={n:float(np.load(ROOT/'results'/f'{n}_probabilities.npy')[index]) for n in names}
    for n in names:assert abs(d['model_probabilities'][n]-expected[n])<3e-5,(n,d['model_probabilities'][n],expected[n])
    assert d['predicted_label']==int(sum(p>=.5 for p in expected.values())>=3)
    assert len(d['heatmap'])>200 and d['explanation']['method']=='Grad-CAM'
    png=base64.b64decode(d['heatmap'].split(',',1)[1]);heat=np.array(Image.open(io.BytesIO(png)))
    assert heat.shape==(224,224,3) and np.std(heat)>0
    im=c.get(URL+chosen['image']).content
    upload=c.post(URL+'/api/predict',files={'file':('xray.png',im,'image/png')});upload.raise_for_status();u=upload.json()
    assert u['predicted_label']==d['predicted_label'] and abs(u['pneumonia_score']-d['pneumonia_score'])<1e-6
    assert 'known_test_label' not in u
    jpeg=io.BytesIO();Image.open(io.BytesIO(im)).convert('RGB').save(jpeg,format='JPEG',quality=95)
    jr=c.post(URL+'/api/predict',files={'file':('xray.jpg',jpeg.getvalue(),'image/jpeg')});jr.raise_for_status();jd=jr.json()
    assert 0<=jd['pneumonia_score']<=1 and len(jd['model_probabilities'])==5
    assert jd['predicted_label']==int(sum(v>=.5 for v in jd['model_probabilities'].values())>=3)
    invalid=c.post(URL+'/api/predict',files={'file':('bad.png',b'not an image','image/png')});assert invalid.status_code==400
    missing=c.post(URL+'/api/predict');assert missing.status_code==400
    unknown=c.get(URL+'/api/sample/not-a-case');assert unknown.status_code==404
    evidence=dict(status='passed',sample_id=chosen['id'],prediction=d['prediction'],score=d['pneumonia_score'],live_probabilities_match_saved_test_outputs=True,upload_and_sample_equivalent=True,jpeg_upload_valid=True,real_gradcam_png_valid=True,invalid_image_status=invalid.status_code,missing_input_status=missing.status_code,unknown_sample_status=unknown.status_code,seconds=d['seconds'],explanation=d['explanation'])
    (ROOT/'results/demo_verification.json').write_text(json.dumps(evidence,indent=2));print(json.dumps(evidence,indent=2))
