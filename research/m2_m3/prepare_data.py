"""Download one verified mirror shard at a time, retain original encoded images."""
import hashlib, io, json, subprocess, time
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'; DATA.mkdir(exist_ok=True)
BASE='https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia/resolve/main/'
def prepare():
    (ROOT/'sources').mkdir(exist_ok=True)
    p=subprocess.run(['curl','-fLsS','--retry','2','--max-time','60','https://huggingface.co/api/datasets/hf-vision/chest-xray-pneumonia/tree/main/data'],capture_output=True,text=True,check=True)
    files=json.loads(p.stdout)
    (ROOT/'sources'/'dataset_files.json').write_text(json.dumps(files,indent=2))
    records=[]
    # The original test is retained for comparison, never used to train/tune.
    for info in files:
        remote=info['path']; split=Path(remote).name.split('-')[0]
        shard=DATA/Path(remote).name
        done=shard.with_suffix('.manifest.json')
        if done.exists(): records.extend(json.loads(done.read_text()));continue
        print('Downloading',remote,info['size']/1e6,'MB',flush=True)
        subprocess.run(['curl','-fLsS','--retry','3','--max-time','900','--limit-rate','8M',BASE+remote,'-o',str(shard)],check=True)
        digest=hashlib.sha256(shard.read_bytes()).hexdigest()
        if digest!=info['lfs']['oid']: raise RuntimeError('Shard checksum mismatch')
        out=[]
        pf=pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=16):
            for row in batch.to_pylist():
                raw=row['image']['bytes']; sha=hashlib.sha256(raw).hexdigest(); label=int(row['label'])
                target=DATA/'images'/split/str(label)/(sha+'.png');target.parent.mkdir(parents=True,exist_ok=True)
                # Cache paper's initial 224x224 RGB resize losslessly to save disk.
                with Image.open(io.BytesIO(raw)) as im:
                    w,h=im.size
                    if not target.exists():im.convert('RGB').resize((224,224),Image.Resampling.BILINEAR).save(target)
                out.append(dict(path=str(target.relative_to(ROOT)),split=split,label=label,sha256=sha,width=w,height=h,source_path=row['image'].get('path')))
        done.write_text(json.dumps(out,indent=2));records.extend(out)
        # Own downloaded cache only; original JPEG bytes and checksums retained.
        shard.unlink()
        print('Verified/extracted',len(out),'images; removed redundant parquet cache',flush=True)
    (DATA/'manifest.json').write_text(json.dumps(records,indent=2))
    print('Manifest records',len(records),'unique files',len({r['sha256'] for r in records}),flush=True)
if __name__=='__main__':prepare()
