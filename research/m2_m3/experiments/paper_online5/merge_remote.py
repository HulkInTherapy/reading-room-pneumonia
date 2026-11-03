"""Merge verified same-protocol remote final checkpoints without overwrites."""
import json,hashlib,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parents[2]/'online5_offload/m2_m3/experiments/paper_online5'
OUT=HERE/'results';OUT.mkdir(exist_ok=True)
local=json.loads((HERE/'PREFLIGHT.json').read_text());remote=json.loads((SOURCE/'PREFLIGHT.json').read_text())
assert local['passed'] and remote['passed'] and local['config_hash']==remote['config_hash']
assert local['source_sha256']==remote['source_sha256']
def copy_new(source,target):
    if target.exists():assert hashlib.sha256(source.read_bytes()).digest()==hashlib.sha256(target.read_bytes()).digest();return
    temporary=target.with_suffix(target.suffix+'.merge_tmp');shutil.copy2(source,temporary);temporary.replace(target)
for name in ['densenet121','inception_v3']:
    complete=json.loads((SOURCE/'results'/f'{name}_complete.json').read_text())
    checkpoint=SOURCE/'results'/f'{name}_final.pt'
    assert complete['epochs']==5 and complete['train_records']==5232 and complete['config_hash']==local['config_hash']
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==complete['checkpoint_sha256']
    history=json.loads((SOURCE/'results'/f'{name}_history.json').read_text())
    assert len(history)==5 and all(h['training_records']==5232 for h in history)
    # Publish the completion marker last.
    for suffix in ['_final.pt','_history.json','_complete.json']:copy_new(SOURCE/'results'/f'{name}{suffix}',OUT/f'{name}{suffix}')
copy_new(SOURCE/'PREFLIGHT.json',HERE/'VPS_PREFLIGHT.json');copy_new(SOURCE/'PLATFORM.json',HERE/'VPS_PLATFORM.json')
print('Remote checkpoints merged. Config, source hashes, epoch counts and file hashes verified.')
