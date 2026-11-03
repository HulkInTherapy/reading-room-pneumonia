"""Assemble and verify the portable model with network connections disabled."""
from pathlib import Path
import csv,json,shutil,hashlib,importlib.util,os,socket,time
P=Path(__file__).resolve().parent;B=P/'model_bundle';O=P/'results';(B/'heads').mkdir(exist_ok=True)
assert json.loads((O/'verification.json').read_text())['verified']
for p in O.glob('*.pt'):shutil.copy2(p,B/'heads'/p.name)
shutil.copy2(O/'summary.json',B/'summary.json');shutil.copy2(P/'requirements-lock.txt',B/'requirements-lock.txt')
rows=list(csv.DictReader((O/'predictions.csv').open()));split=json.loads((P/'split.json').read_text());cases=[(i,split['test_records'][i]['sha256']) for i in [0,234]]
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
def deny(*args,**kwargs):raise RuntimeError('Network connection attempted during portable verification')
socket.socket.connect=deny;socket.socket.connect_ex=deny
spec=importlib.util.spec_from_file_location('portable_model',B/'inference.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from PIL import Image
checks=[]
for i,sha in cases:
 full=Image.open(P/'data/full'/(sha+'.png')).convert('RGB');crop=Image.open(P/'data/crop'/(sha+'.png')).convert('RGB');started=time.time();result=m.classify(full,crop);seconds=time.time()-started;expected=rows[i];assert expected['sha256']==sha
 delta={k:abs(float(expected[k])-result['conditions'][k]['pneumonia_score']) for k in 'ABCD'};assert max(delta.values())<2e-5,delta
 example=B/'examples'/('normal' if i==0 else 'pneumonia');example.mkdir(parents=True,exist_ok=True)
 for kind in ['full','crop','masks']:shutil.copy2(P/'data'/kind/(sha+'.png'),example/(kind+'.png'))
 result.update(sample_index=i,source_sha256=sha,known_label=int(expected['label']),seconds=seconds);(example/'prepared_prediction.json').write_text(json.dumps(result,indent=2));checks.append(dict(sample_index=i,max_probability_delta=delta,seconds=seconds))
# Prove that the included segmentation weights also load with networking forbidden.
full=Image.open(B/'examples/normal/full.png').convert('RGB');f,c,mask,meta=m.segment(full);assert mask.size==(256,256);mask.save(B/'examples/normal/resegmented_mask_from_224_input.png')
report=dict(verified=True,network_connections_disabled=True,prepared_case_checks=checks,segmenter_loaded_offline=True,segmenter_input_fixture='224px exported example, not original-resolution source',segmentation_fixture_metadata=meta)
(O/'portable_verification.json').write_text(json.dumps(report,indent=2));(B/'verification.json').write_text(json.dumps(report,indent=2))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
manifest={str(p.relative_to(B)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in B.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.name!='MODEL_MANIFEST.json'}
(B/'MODEL_MANIFEST.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(report,indent=2))
