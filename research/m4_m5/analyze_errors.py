import csv,json
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
R=Path(__file__).resolve().parent;O=R/'results';O.mkdir(exist_ok=True)
rows=list(csv.DictReader((R.parent/'m2_m3/experiments/paper_online5/results/test_predictions.csv').open()))
errors=[r for r in rows if r['true_label']!=r['predicted_label']]
chosen=[r for r in errors if r['true_label']=='0'][:4]+[r for r in errors if r['true_label']=='1'][:4]
fig,ax=plt.subplots(2,4,figsize=(12,6.5),facecolor='#f5f4ef')
for a,r in zip(ax.ravel(),chosen):
 a.imshow(Image.open(R.parent/'m2_m3'/r['path']),cmap='gray');a.axis('off');a.set_title(('False positive' if r['true_label']=='0' else 'False negative')+'\nPneumonia score '+format(float(r['mean_score']),'.2f'),fontsize=10)
fig.tight_layout();fig.savefig(O/'baseline_errors.png',dpi=150);plt.close(fig)
(O/'baseline_error_examples.json').write_text(json.dumps(chosen,indent=2))
