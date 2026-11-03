from pathlib import Path
import zipfile,json,hashlib
ROOT=Path('/Users/zebra/cvproj/m2_m3')
out=ROOT/'output'
include=['baseline.py','prepare_data.py','audit.py','verify_results.py','test_boundaries.py','test_demo.py','requirements.txt','launch_demo.py','START_DEMO.command','DEMO_PRESENTATION_GUIDE.md','HYPOTHESIS_BEFORE_IMPLEMENTATION.md','INTRODUCTION.md','ASSIGNMENT_AND_PLAN.md','REPRODUCIBILITY_CHECKLIST.md','SOURCE_AUDIT.md','README.md']
files=[ROOT/f for f in include]+list(out.glob('*.ipynb'))+list(out.glob('*.pptx'))+list(out.glob('*.pdf'))+list(out.glob('*.docx'))+[out/'TEAM_SHARED_PREP.md']
files+=list((ROOT/'results').glob('*.json'))+list((ROOT/'results').glob('*.csv'))+list((ROOT/'results').glob('*.png'))
files+=list((ROOT/'results').glob('*_probabilities.npy'))
files+=[ROOT/'data/manifest.json',ROOT/'sources/pretrained_weight_checksums.json',ROOT/'sources/dataset_files.json']
files+=list((ROOT/'demo').glob('*.py'))+[ROOT/'demo/index.html']+list((ROOT/'demo/assets').glob('*'))
exp=ROOT/'experiments/restart15'
files+=list(exp.glob('*.py'))+list(exp.glob('*.json'))+list(exp.glob('*.md'))+list(exp.glob('*.txt'))
for extension in ['json','csv','npy','png','html']:
 files+=list((exp/'results').glob('*.'+extension))
exp=ROOT/'experiments/paper_online5'
files+=list(exp.glob('*.py'))+list(exp.glob('*.json'))+list(exp.glob('*.txt'))
for extension in ['json','csv','npy','png','html']:
 files+=list((exp/'results').glob('*.'+extension))
manifest=ROOT/'results/deliverable_checksums.json'
files=[f for f in files if f!=manifest]
checksums={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
manifest.write_text(json.dumps(checksums,indent=2))
files.append(manifest)
with zipfile.ZipFile(out/'M2_M3_Submission_Package.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in files:
  if not f.exists():raise RuntimeError(f'Missing promised deliverable {f}')
  z.write(f,'M2_M3/'+str(f.relative_to(ROOT)))
print('Submission ZIP excludes large images, feature caches, learned head weights and virtualenv.')
print(out/'M2_M3_Submission_Package.zip')
