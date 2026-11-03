import json, collections, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def audit():
    rows=json.loads((ROOT/'data/manifest.json').read_text())
    counts={s:{str(c):sum(r['split']==s and r['label']==c for r in rows) for c in [0,1]} for s in ['train','validation','test']}
    train=[r for r in rows if r['split']!='test'];test=[r for r in rows if r['split']=='test']
    tr={r['sha256'] for r in train};te={r['sha256'] for r in test}
    out=dict(counts=counts,total_records=len(rows),unique_source_images=len({r['sha256'] for r in rows}),train_records=len(train),test_records=len(test),unique_train=len(tr),unique_test=len(te),exact_hash_overlap=len(tr&te),source='hf-vision/chest-xray-pneumonia',source_dataset='10.17632/rscbjbr9sj.2',manifest_sha256=hashlib.sha256((ROOT/'data/manifest.json').read_bytes()).hexdigest(),cache='RGB bilinear initial resize 224x224, lossless PNG',test_majority_accuracy=.625)
    assert out['exact_hash_overlap']==0
    (ROOT/'results').mkdir(exist_ok=True);(ROOT/'results/dataset_audit.json').write_text(json.dumps(out,indent=2))
    return out
if __name__=='__main__':print(json.dumps(audit(),indent=2))
