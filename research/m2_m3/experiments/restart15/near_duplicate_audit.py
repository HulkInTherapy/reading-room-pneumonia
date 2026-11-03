"""Read-only perceptual screening; no split or model selection changes."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import json,time
import numpy as np
from scipy.fft import dctn
from PIL import Image
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
s=json.loads((HERE/'split.json').read_text());rows=s['records']+s['test_records']
hashes=[];thumbnails=[]
for r in rows:
    with Image.open(ROOT/r['path']) as im:
        a=np.array(im.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=np.float32)/255
    low=dctn(a,norm='ortho')[:8,:8].flatten()[1:]
    hashes.append(np.packbits(low>np.median(low)));thumbnails.append(a)
hashes=np.array(hashes);thumbnails=np.array(thumbnails)
popcount=np.array([int(i).bit_count() for i in range(256)],dtype=np.uint8)
def compare(left,right):
    candidates=[];screened=0
    for k in range(0,len(left),128):
        ix=np.array(left[k:k+128]);dist=popcount[np.bitwise_xor(hashes[ix,None,:],hashes[np.array(right)][None,:,:])].sum(2)
        for a,b in np.argwhere(dist<=4):
            i,j=int(ix[a]),int(right[b]);screened+=1
            mae=float(np.mean(np.abs(thumbnails[i]-thumbnails[j])))
            if mae<=.03:candidates.append(dict(left_path=rows[i]['path'],right_path=rows[j]['path'],left_label=rows[i]['label'],right_label=rows[j]['label'],phash_distance=int(dist[a,b]),thumbnail_mean_absolute_difference=mae,exact_source_hash=rows[i]['sha256']==rows[j]['sha256']))
        time.sleep(.05)
    return dict(low_phash_distance_pairs=screened,low_pixel_difference_candidates=candidates)
test=list(range(len(s['records']),len(rows)))
result=dict(method='63-bit DCT perceptual hash distance <=4, followed by normalized 32x32 grayscale MAE <=0.03',warning='Approximate resized-image screen, not proof of duplicate image or patient identity. Negative screening does not prove independence. No split/checkpoint/threshold changes are made from this audit.',train_validation=compare(s['train_indices'],s['validation_indices']),non_test_test=compare(list(range(len(s['records']))),test))
(HERE/'near_duplicate_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:len(result[k]['low_pixel_difference_candidates']) for k in ['train_validation','non_test_test']}))
