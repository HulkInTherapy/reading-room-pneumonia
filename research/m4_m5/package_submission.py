"""Package explicit final deliverables without environments, secrets or caches."""
from pathlib import Path
import zipfile,hashlib,json
R=Path(__file__).resolve().parent;O=R/'output';s=json.loads((R/'results/summary.json').read_text());assert s['completed'];assert json.loads((R/'results/verification.json').read_text())['verified']
files=[]
for pattern in ['output/*.pdf','output/*.html','output/*.ipynb','output/*.json','output/*.txt','results/*.json','results/*.csv','results/*.png','results/*_history.json']:
 files.extend(R.glob(pattern))
files.extend(p for p in R.glob('*.py'));files.extend([R/'START_FINAL_DEMO.command',R/'PRESENTATION_RUNBOOK.txt'])
files.extend((R.parent/'m2_m3/demo').glob('*.py'));files.append(R.parent/'m2_m3/demo/index.html')
files.extend([R.parent/'m2_m3/data/manifest.json',R.parent/'m2_m3/experiments/restart15/split.json',R.parent/'m2_m3/output/M2_Final_Online_Baseline.ipynb'])
for name in ['summary.json','test_predictions.csv','independent_verification.json']:
 files.append(R.parent/'m2_m3/experiments/paper_online5/results'/name)
files.extend([R.parent/'deck/M1-Literature-Survey-Pneumonia-CXR.pdf',R.parent/'m2_m3/output/M2_M3_Project_Report.docx']);files.extend([R.parent/'FINAL_START_HERE.txt',R.parent/'m2_m3/README.md',R.parent/'m2_m3/HYPOTHESIS_BEFORE_IMPLEMENTATION.md',R.parent/'m2_m3/REPRODUCIBILITY_CHECKLIST.md']);files.extend([R/'runtime_report.json',R/'status.json',R/'train.log',R/'train256.log']);files.extend(R.glob('requirements*'));files.extend(R.glob('*freeze*'));files.extend(R.glob('*benchmark.json'));files.extend(R.glob('*protocol*'))
files=list(dict.fromkeys(p for p in files if p.exists() and p.name!='partial_summary.json'))
checks={str(p.relative_to(R.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(O/'SUBMISSION_CHECKSUMS.json').write_text(json.dumps(checks,indent=2))
with zipfile.ZipFile(O/'Final_M4_M5_Submission.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in files:z.write(p,str(p.relative_to(R.parent)))
 z.write(O/'SUBMISSION_CHECKSUMS.json','m4_m5/output/SUBMISSION_CHECKSUMS.json')
with zipfile.ZipFile(O/'Trained_Pilot_Heads.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in (R/'results').glob('*.pt'):z.write(p,p.name)
 z.write(R/'results/summary.json','summary.json')
 z.write(R.parent/'m2_m3/demo/pilot.py','inference/pilot.py')
 z.writestr('README.txt','These are the four trained five-epoch classifier heads and training-feature normalization statistics. They require the fixed ImageNet ResNet18 and DeiT-Tiny encoders, plus the pretrained PSPNet segmenter for raw uploads. The complete prepared model server runs on our VPS. Refer to the project runbook and exact environment. This is a frozen-feature pilot, not the original five-CNN late-block system.\n')
print('Submission files',len(files))

bundle=R/'model_bundle'
required=['weights/pspnet_chestxray_best_model_4.pth','weights/resnet18-f37072fd.pth','weights/deit_tiny_encoder.pt','summary.json','inference.py']+[f'heads/{arch}_{kind}.pt' for arch in ['resnet18','deit_tiny'] for kind in ['full','crop']]
if all((bundle/f).exists() for f in required):
 with zipfile.ZipFile(O/'Portable_Lung_Model.zip','w',zipfile.ZIP_STORED) as z:
  for p in bundle.rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.py','.pt','.pth','.json','.txt','.png']:
    z.write(p,str(p.relative_to(bundle)))
 print('Portable complete model package built')
else:raise SystemExit('Portable model weights incomplete: '+str([f for f in required if not (bundle/f).exists()]))
