import json
from pathlib import Path
from PIL import Image
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;case=json.loads((R/'results/fallback_examples.json').read_text())['cases'][0];sha=case['sha256']
fig,axes=plt.subplots(1,3,figsize=(12,4))
for ax,kind,title in zip(axes,['full','masks','crop'],['Full X-ray','Partial predicted lung mask','Full-image fallback']):
 ax.imshow(Image.open(R/'data'/kind/(sha+'.png')),cmap='gray');ax.axis('off');ax.set_title(title,fontsize=15)
fig.tight_layout();fig.savefig(R/'results/fallback_pipeline.png',dpi=180);plt.close(fig)
