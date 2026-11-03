"""Check deployed inference against fixed predictions and exercise upload guards."""
from pathlib import Path
import json,csv,io,time,base64,math
import httpx
from PIL import Image
P=Path(__file__).resolve().parent;O=P/'results';rows=list(csv.DictReader((O/'predictions.csv').open()));report={};client=httpx.Client(base_url='http://127.0.0.1:8765',timeout=120)
for _ in range(30):
 try:
  status=client.get('/api/status');status.raise_for_status();break
 except httpx.HTTPError:time.sleep(1)
s=client.get('/api/results').json();assert s['experiments']['completed'] and s['verification']['verified']
for index,label in [(0,'normal'),(234,'pneumonia')]:
 r=client.post('/api/experiment/predict',data={'sample_id':str(index)});assert r.status_code==200,r.text;d=r.json();expected=rows[index];delta={k:abs(float(expected[k])-d['conditions'][k]['pneumonia_score']) for k in 'ABCD'};assert max(delta.values())<2e-5,delta
 assert d['preprocessing_seconds']>=0 and d['classifier_seconds']>0 and d['seconds']>=d['preprocessing_seconds']+d['classifier_seconds'];(O/f'demo_pilot_{label}_recorded.json').write_text(json.dumps(d,indent=2));report[label]=dict(status_code=r.status_code,max_probability_difference=delta,seconds=d['seconds'],preprocessing_seconds=d['preprocessing_seconds'],classifier_seconds=d['classifier_seconds'],timing_context='Server-side processing; cached prepared sample masks, includes encoder/head loads and image encoding, excludes HTTP transport')
sha=rows[0]['sha256'];image=P/'data/full'/(sha+'.png')
with image.open('rb') as f:r=client.post('/api/experiment/predict',files={'file':('prepared_full_224.png',f,'image/png')})
assert r.status_code==200,r.text;d=r.json()
for k in 'ABCD':assert math.isfinite(d['conditions'][k]['pneumonia_score']) and 0<=d['conditions'][k]['pneumonia_score']<=1
mask=Image.open(io.BytesIO(base64.b64decode(d['mask_image'].split(',',1)[1])));assert mask.size==(256,256);assert d['crop_metadata']['original_size']==[224,224];(O/'demo_pilot_upload_recorded.json').write_text(json.dumps(d,indent=2));report['upload']=dict(status_code=200,seconds=d['seconds'],preprocessing_seconds=d['preprocessing_seconds'],classifier_seconds=d['classifier_seconds'],timing_context='Server-side uploaded224px fixture; includes PSPNet load/segmentation, encoder/head loads and PNG encoding; excludes HTTP upload transport')
checks={}
for name,kwargs,expected in [('unreadable',{'files':{'file':('bad.png',b'not an image','image/png')}},400),('unknown_sample',{'data':{'sample_id':'missing'}},400),('oversize_bytes',{'files':{'file':('large.png',b'0'*(12*1024**2+1),'image/png')}},413)]:
 r=client.post('/api/experiment/predict',**kwargs);assert r.status_code==expected,(name,r.status_code,r.text);checks[name]=r.status_code
im=Image.new('L',(6000,5000));b=io.BytesIO();im.save(b,format='PNG');r=client.post('/api/experiment/predict',files={'file':('oversize_dimensions.png',b.getvalue(),'image/png')});assert r.status_code==413,r.text;checks['oversize_dimensions']=413
report['invalid_upload_checks']=checks;report['verified']=True;(O/'api_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
