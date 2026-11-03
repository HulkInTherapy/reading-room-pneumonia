"""Build PDF slides, an offline HTML presentation, and a two-column final paper."""
import json,html,base64,textwrap
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.platypus import BaseDocTemplate,PageTemplate,Frame,Paragraph,Spacer,Image as PDFImage,FrameBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_CENTER
R=Path(__file__).resolve().parent; O=R/'output';O.mkdir(exist_ok=True)
baseline=json.loads((R.parent/'m2_m3/experiments/paper_online5/results/summary.json').read_text())
p=R/'results/summary.json'
if not p.exists():raise SystemExit('Waiting for measured experiment summary; no final materials generated.')
e=json.loads(p.read_text());
if not e.get('completed',False):raise SystemExit('Experiment not verified complete')
vpath=R/'results/verification.json'
if not vpath.exists() or not json.loads(vpath.read_text()).get('verified',False):raise SystemExit('Waiting for independent verification')
(O/'measured_results.json').write_text(json.dumps(e,indent=2))
conditions=e['conditions']; print(json.dumps(e,indent=2)[:2000])
rows=[]
for k,v in conditions.items():
 m=v.get('metrics',v.get('test_metrics',{}));rows.append(f"{k}: accuracy {m.get('accuracy',0):.2%}, recall {m.get('sensitivity',0):.2%}, specificity {m.get('specificity',0):.2%}, macro F1 {m.get('macro_f1',0):.3f}, AUC {m.get('auc',0):.3f}")
slides=[
('Reading the Reading Room',['Pneumonia classification from pediatric chest X-rays','Final milestones: anatomy-guided inputs and matched classifier comparisons','Samarth Shekhar · Meghavi Rao · Rachit Kumar · Harshal Nerpagar']),
('From reproduction to a controlled experiment',['M1: literature review and dataset selection','M2: reproduced five-CNN baseline and audited predictions','M3: predicted gains from lung cropping and a pretrained transformer','M4–M5: implementation, measured comparisons and failure analysis']),
('Dataset and evaluation',['Normal versus pneumonia on the Kermany pediatric benchmark','Historical baseline: 5,232 training records and 624 test records','New split: 4,164 training, 1,042 validation and 624 test records','The benchmark test was inspected previously. It is not an external blind holdout.']),
('The verified M2 baseline',['Five pretrained CNNs, five epochs, majority voting. Paper target: 96.4% accuracy.','Accuracy 92.15% · pneumonia recall 94.62% · specificity 88.03%','Macro F1 0.916 · ROC AUC 0.962','28 false positives, 21 false negatives. ResNet18 alone: 93.11% accuracy.']),
('Why focus on the lungs?',['The full image includes peripheral anatomy and acquisition markings','A fixed pretrained anatomical model predicts both lungs at 256 pixels','We crop their union with a 10% margin, then resize for classification','256 replaces the standard 512 input for CPU runtime. Failed masks fall back.']),
('The implementation and its limits',['Frozen pretrained image encoders with five-epoch trained classifier heads','ResNet18 supplies CNN features. DeiT-Tiny supplies transformer features.','The pilot isolates image region and classifier combination under one split','This reduced experiment differs from the original five-CNN late-block proposal.',
'Uploaded 224 px fixture: 4.58 s including segmentation; cached cases: 1.24–1.49 s.']),
('Four matched conditions',['A: full image with ResNet18','B: lung-focused crop with ResNet18','C: full image with ResNet18 and DeiT-Tiny','D: lung-focused crop with ResNet18 and DeiT-Tiny','Checkpoints and combination weights use validation data only.']),
('Measured results',rows),
('Interpreting the comparison',['B versus A tests the crop effect for the CNN pilot','C versus A tests the transformer addition on full images','D versus B tests the transformer addition on crops','The historical M2 score uses a different training protocol and is context only.']),
('Errors and uncertainty',['Inspect false positives, false negatives and segmentation failures','Report sensitivity alongside specificity and macro F1','Paired uncertainty estimates describe this benchmark, not other hospitals','Grad-CAM highlights one CNN response. It does not establish lesion boundaries.']),
('Live demonstration',['Select a benchmark example or upload a PNG/JPEG','Inspect the model prediction, individual scores and Grad-CAM','Inspect lung crop experiment metrics and recorded protocol','The interface identifies the model being demonstrated and uses saved trained weights.']),
('Conclusions and remaining questions',['Predeclared crop CNN–ViT: 93.75% accuracy and 98.21% pneumonia recall','Small pediatric data and imperfect identifier groups limit generalization','Adding ViT to the crop CNN improved accuracy by 1.12 percentage points','The validation/test reversal requires repeated splits and external testing.'])]
verification=json.loads(vpath.read_text());selection=verification['validation_only_selection_audit']
val={}
for key,base in [('C','A'),('D','B')]:
 grid=selection[key]['grid'];weight=selection[key]['selected_vit_weight']
 val[base]=next(r['validation_macro_f1'] for r in grid if r['vit_weight']==0)
 val[key]=next(r['validation_macro_f1'] for r in grid if r['vit_weight']==weight)
slides.insert(9,('Validation and test disagree on cropping',[
 f"A: validation macro F1 {val['A']:.3f}; test {conditions['A']['metrics']['macro_f1']:.3f}",
 f"B: validation macro F1 {val['B']:.3f}; test {conditions['B']['metrics']['macro_f1']:.3f}",
 f"C: validation macro F1 {val['C']:.3f}; test {conditions['C']['metrics']['macro_f1']:.3f}",
 f"D: validation macro F1 {val['D']:.3f}; test {conditions['D']['metrics']['macro_f1']:.3f}",
 'This single split supports a benchmark result, not universal crop superiority.']))
audit_path=R/'results/preprocessing_audit.json'
if audit_path.exists():
 crop_audit=json.loads(audit_path.read_text())
 for title,lines in slides:
  if title=='Errors and uncertainty':
   lines.insert(0,f"Automatic full-image fallbacks: {crop_audit['automatic_fallbacks']} / {crop_audit['unique_images']} ({crop_audit['automatic_fallbacks']/crop_audit['unique_images']:.2%})")
  if title=='Why focus on the lungs?':
   lines.append(f"Median retained crop area: {crop_audit['median_crop_area_fraction']:.1%} of original image area")
a=conditions['A']['metrics'];b=conditions['B']['metrics']
ci=e.get('paired_macro_f1_B_minus_A_95CI')
ci_text=('['+format(ci[0],'.4f')+', '+format(ci[1],'.4f')+']') if ci is not None else 'not available'
delta=b['macro_f1']-a['macro_f1'];specificity=100*(b['specificity']-a['specificity'])
if delta<=0:verdict='The measured crop effect does not support the predicted macro-F1 improvement.'
elif delta<.01:verdict='Macro F1 improves, but less than the predicted +0.01 to +0.03.'
elif delta<=.03:verdict='The measured macro-F1 gain meets the predicted +0.01 to +0.03 range.'
else:verdict='The measured macro-F1 gain exceeds the predicted +0.03 upper bound.'
if ci is not None and ci[0]<=0<=ci[1]:verdict+=' The paired interval includes zero.'
slides.insert(8,('What happened to the M3 prediction?',[
 f'Crop effect B minus A: macro F1 {delta:+.4f}, specificity {specificity:+.2f} percentage points',
 'Paired 95% interval for macro-F1 difference: '+ci_text,verdict,
 'Validation-selected ViT weights: C='+str(conditions['C'].get('vit_weight'))+', D='+str(conditions['D'].get('vit_weight'))]))
# Add measured visual evidence when verification has produced it.
visuals={}
for title,filename in [('Confusion matrices by condition','confusion_matrices.png'),('Training examples from both classes','training_balanced_mask_sanity.png'),('Historical baseline error examples','baseline_errors.png'),('The crop pipeline on a training image','segmentation_pipeline.png'),('Where the crop hybrid makes errors','pilot_errors.png'),('When the pipeline falls back to the full image','fallback_pipeline.png')]:
 path=R/'results'/filename
 if path.exists():
  slides.insert(-2,(title,[]));visuals[title]=path
slides.append(('References',[
'Chouhan et al. (2020). Five-CNN transfer-learning ensemble. doi:10.3390/app10020559',
'Cohen et al. (2022). TorchXRayVision. mlmed.org/torchxrayvision/models.html',
'Touvron et al. (2021). Data-efficient image transformers. github.com/facebookresearch/deit',
'Abdullah et al. (2026). HybridCATNet. doi:10.3389/fdgth.2026.1769444',
'Kermany et al. Pediatric chest X-ray dataset. doi:10.17632/rscbjbr9sj.2']))
main_order=['Reading the Reading Room','From reproduction to a controlled experiment','Dataset and evaluation','The verified M2 baseline','Why focus on the lungs?','The crop pipeline on a training image','The implementation and its limits','Four matched conditions','Measured results','Confusion matrices by condition','What happened to the M3 prediction?','Validation and test disagree on cropping','Errors and uncertainty','When the pipeline falls back to the full image','Live demonstration','Conclusions and remaining questions']
by_title=dict(slides);ordered=[(t,by_title[t]) for t in main_order if t in by_title]
for title,lines in slides:
 if title not in main_order:
  new_title=title if title=='References' else 'Appendix: '+title
  ordered.append((new_title,lines))
  if title in visuals:visuals[new_title]=visuals[title]
slides=ordered
(O/'slides.json').write_text(json.dumps(slides,indent=2))
# Vector PDF slides. Keep text sparse, 16:9, readable at projection size.
c=canvas.Canvas(str(O/'Final_M4_M5_Presentation.pdf'),pagesize=(960,540))
for i,(title,lines) in enumerate(slides):
 c.setFillColor(HexColor('#f5f4ef'));c.rect(0,0,960,540,fill=1,stroke=0)
 c.setFillColor(HexColor('#132736'));c.setFont('Helvetica-Bold',29)
 for j,line in enumerate(textwrap.wrap(title,48)):c.drawString(48,477-j*36,line)
 y=382
 if title=='Measured results':
  xs=[67,210,350,490,670,800]
  c.setFillColor(HexColor('#dfebea'));c.rect(48,368,864,44,fill=1,stroke=0)
  c.setFillColor(HexColor('#132736'));c.setFont('Helvetica-Bold',17)
  for x,value in zip(xs,['Run','Accuracy','Recall','Specificity','Macro F1','AUC']):c.drawString(x,384,value)
  for row,key in enumerate('ABCD'):
   m=conditions[key]['metrics'];yy=334-row*54
   c.setFillColor(HexColor('#246d73') if key=='D' else HexColor('#243b49'));c.setFont('Helvetica-Bold' if key=='D' else 'Helvetica',20)
   values=[key,format(m['accuracy'],'.2%'),format(m['sensitivity'],'.2%'),format(m['specificity'],'.2%'),format(m['macro_f1'],'.3f'),format(m['auc'],'.3f')]
   for x,value in zip(xs,values):c.drawString(x,yy,value)
  c.setFont('Helvetica',15);c.drawString(48,104,'A: full CNN. B: crop CNN. C: full CNN + ViT. D: crop CNN + ViT.')
  c.drawString(48,78,'624 test records. Validation-selected ViT weights: C 0.25; D 0.50.')
 for line in ([] if title=='Measured results' else lines):
  c.setFillColor(HexColor('#246d73'));c.rect(48,y-3,4,18,fill=1,stroke=0)
  c.setFillColor(HexColor('#243b49'));c.setFont('Helvetica',19)
  for chunk in textwrap.wrap(line,83):c.drawString(67,y,chunk.replace('→','>'));y-=27
  y-=18
 if title in visuals:c.drawImage(str(visuals[title]),48,68,width=864,height=345,preserveAspectRatio=True,anchor='c',mask='auto')
 c.setFillColor(HexColor('#72818a'));c.setFont('Helvetica',10);c.drawString(48,24,'Reading the Reading Room / final milestones');c.drawRightString(912,24,f'{i+1:02} / {len(slides)}')
 c.showPage()
c.save()
sections=[]
for title,lines in slides:sections.append('<section><h1>'+html.escape(title)+'</h1>'+''.join('<p>'+html.escape(t)+'</p>' for t in lines)+(('<img style="max-width:90%;max-height:65vh" src="data:image/png;base64,'+base64.b64encode(visuals[title].read_bytes()).decode()+'">') if title in visuals else '')+'</section>')
(O/'Final_M4_M5_Presentation.html').write_text('''<!doctype html><html><meta charset="utf-8"><title>Reading Room / final milestones</title><style>body{margin:0;background:#f5f4ef;color:#132736;font:24px Arial}section{display:none;box-sizing:border-box;padding:5vw 7vw;height:100vh}section.on{display:block}h1{font-size:44px;margin:0 0 7vh}p{max-width:1100px;line-height:1.5;margin:22px 0}nav{position:fixed;bottom:16px;right:24px;font-size:14px}button{font:inherit;padding:10px 18px;border:1px solid #bbb;background:white;cursor:pointer}@media print{section{display:block;break-after:page;height:100vh}nav{display:none}}</style>'''+''.join(sections)+'''<nav><button onclick="go(-1)">Previous</button> <span id="n"></span> <button onclick="go(1)">Next</button> <button onclick="document.documentElement.requestFullscreen()">Fullscreen</button></nav><script>let i=0,s=[...document.querySelectorAll('section')];function go(d){i=Math.max(0,Math.min(s.length-1,i+d));s.forEach((e,j)=>e.classList.toggle('on',i===j));document.getElementById('n').textContent=(i+1)+' / '+s.length}document.onkeydown=e=>{if(['ArrowRight',' ','PageDown'].includes(e.key)){e.preventDefault();go(1)}if(['ArrowLeft','PageUp'].includes(e.key))go(-1)};go(0)</script></html>''')
styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='PaperBody',fontName='Times-Roman',fontSize=10,leading=12,spaceAfter=6));styles.add(ParagraphStyle(name='PaperHead',fontName='Times-Bold',fontSize=10.5,leading=13,spaceBefore=9,spaceAfter=5,keepWithNext=True))
body=[]
def para(t):body.append(Paragraph(html.escape(t),styles['PaperBody']))
def head(t):body.append(Paragraph(t,styles['PaperHead']))
paper_title='Lung-Focused Inputs for CNN-ViT Pneumonia Classification'
head('Abstract')
para('We reconstruct a five-CNN pneumonia classifier and evaluate an anatomy-guided extension using a resource-bounded frozen-feature pilot. The historical five-epoch ensemble attained 92.15% accuracy and ROC AUC 0.9623 on 624 benchmark records. The final experiment compares full and lung-focused images with CNN-only and CNN-plus-transformer classifiers using validation-selected settings. We report all measured conditions, including unfavorable outcomes, rather than assuming anatomical cropping improves classification.')
para('In the final matched pilot, full-image CNN accuracy was '+format(conditions['A']['metrics']['accuracy'],'.2%')+', lung-crop CNN accuracy was '+format(conditions['B']['metrics']['accuracy'],'.2%')+', and the crop CNN/ViT candidate accuracy was '+format(conditions['D']['metrics']['accuracy'],'.2%')+'. The measured crop macro-F1 change was '+format(delta,'+.4f')+'.')
head('I. Introduction and Related Work')
para('Pediatric pneumonia recognition from radiographs is a binary benchmark task with important limitations in generalization. Chouhan et al. combined five ImageNet CNNs [1]. TorchXRayVision provides pretrained anatomical segmentation [2]. DeiT supplies pretrained data-efficient image transformers [3]. Anatomical CNN–Transformer hybrids already exist [4]. Our contribution is a controlled student reproduction and pilot evaluation, not a new clinical method.')
head('II. Dataset and Historical Reconstruction')
para('We use the public pediatric chest X-ray dataset [5]. The prior reconstruction trained five CNNs on 5,232 records for five epochs. Majority voting produced 206 true negatives, 28 false positives, 21 false negatives and 369 true positives on 624 records. The corresponding sensitivity was 94.62%, specificity 88.03%, precision 92.95% and macro F1 0.9157. This schedule is shorter than the selected paper. Fresh augmentation was already part of this baseline.')
head('III. Final Experiment')
para('The primary hypothesis proposed macro-F1 improvement of 0.01–0.03 and specificity improvement of 2–5 percentage points from anatomical cropping. A separate transformer extension tests complementary predictions. Segmentation uses 256-pixel input rather than the standard 512-pixel wrapper to bound CPU runtime. Its mask defines a crop on original source pixels. The pilot uses 4,164 training and 1,042 validation records, with 624 original test records retained. It uses frozen pretrained encoders and five-epoch classifier-head optimization, which differs from online late-block training in M2. Full and cropped conditions share a split and comparable head training. Validation data select checkpoints and mixtures. Benchmark test predictions do not select settings.')
para('All head experiments use CPU FP32 with two threads and seed 42. AdamW optimizes unweighted cross-entropy for five epochs at learning rate 0.001, weight decay 0.01 and batch size 32. Feature standardization uses training-only means and standard deviations, with a 0.001 standard-deviation floor. Encoder inputs use ImageNet channel normalization. Checkpoint ties favor the earlier epoch; no fresh image augmentation is used in this declared frozen-feature pilot.')
head('IV. Results')
for row in rows:para(row)
unique=e['unique_image_sensitivity_analysis']['D']
para('On the 618 unique test images, D achieved '+format(unique['accuracy'],'.2%')+' accuracy, '+format(unique['macro_f1'],'.4f')+' macro F1 and '+format(unique['auc'],'.4f')+' AUC. The selected ViT probability weight was 0.25 for C and 0.50 for D; the remaining weight belongs to the corresponding CNN. Labels use a fixed 0.5 threshold.')
para(verdict+' Crop macro-F1 difference B minus A: '+str(round(delta,5))+'. Paired interval: '+ci_text+'. Specificity change: '+str(round(specificity,3))+' percentage points.')
runtime_path=R/'runtime_report.json'
if runtime_path.exists():
 runtime=json.loads(runtime_path.read_text())
 para('The measured interval between the first and last uniform 256 px mask files was '+format(runtime['mask_file_wall_interval_seconds']/60,'.1f')+' minutes. This includes inter-image pauses and later shard downloads but excludes the initial model and first-shard downloads. Head-training time was not independently captured.')
 para('Classifier parameter counts (excluding the fixed segmenter): '+', '.join(k+': '+format(v,',') for k,v in runtime['parameter_counts'].items())+'.')
para('Validation macro F1 was '+', '.join(k+' '+format(val[k],'.4f') for k in 'ABCD')+'. Cropping reduced validation macro F1 despite improving benchmark test macro F1. The direction reverses across these partitions. This single-split experiment therefore does not establish robust crop superiority; the cause of this reversal is unresolved. D was the predeclared target condition, not a winner selected across conditions using test scores.')
head('V. Failure Analysis and Limitations')
if audit_path.exists():
 para('Automatic geometry and empty-mask checks used the full-image fallback for '+str(crop_audit['automatic_fallbacks'])+' of '+str(crop_audit['unique_images'])+' unique images ('+format(crop_audit['automatic_fallbacks']/crop_audit['unique_images'],'.2%')+'). The median retained crop area was '+format(crop_audit['median_crop_area_fraction'],'.1%')+' of the original area. This is an automatic plausibility audit, not a measured segmentation error rate: manual mask annotations are unavailable.')
para('False positives and false negatives should be assessed together. Anatomical crops may exclude relevant tissue, and pretrained segmentation may transfer imperfectly to pediatric images. Filename groups are imperfect patient identifiers. Duplicate-image sensitivity analysis and paired intervals improve transparency but do not establish external generalization. The benchmark test was inspected during prior work and is not a blind external holdout. Uncalibrated probability scores are not clinical risk estimates. Grad-CAM provides qualitative component attribution without lesion ground truth. The reduced pilot does not establish performance of the original full five-CNN-plus-ViT proposal.')
portable=json.loads((R/'results/portable_verification.json').read_text())
segmenter_count=portable['segmentation_fixture_metadata']['segmenter_parameters']
para('The fixed anatomical segmenter contains '+format(segmenter_count,',')+' parameters. Including the two encoders and two D heads, the complete D pipeline contains 84,770,656 neural-network parameters; only 1,412 classifier-head parameters are optimized for D. The segmenter is therefore a substantial part of total compute, rather than a free preprocessing operation.')
api=json.loads((R/'results/api_verification.json').read_text())
para('Measured server-side processing for two cached prepared samples was '+format(api['normal']['seconds'],'.2f')+' and '+format(api['pneumonia']['seconds'],'.2f')+' seconds. One uploaded 224-pixel fixture took '+format(api['upload']['seconds'],'.2f')+' seconds, including '+format(api['upload']['preprocessing_seconds'],'.2f')+' seconds of preprocessing and '+format(api['upload']['classifier_seconds'],'.2f')+' seconds of classifier processing. Timings include model loads and image encoding where applicable and exclude HTTP transport. These are fixture measurements, not latency guarantees.')
head('VI. Conclusion')
para('The predeclared crop CNN/ViT condition achieved 93.75% accuracy, 98.21% sensitivity and AUC 0.9757 on 624 benchmark records. The primary crop macro-F1 gain exceeded the predicted range, while the validation reversal limits generalization. The project delivers a runnable model and a reproducible resource-bounded comparison; repeated patient-aware splits and independent external data remain necessary.')
body.append(FrameBreak())
body.append(PDFImage(str(R/'results/roc.png'),width=250,height=187.5))
para('Figure 1. ROC curves for the four declared conditions on 624 test records.')
head('References')
for t in ['[1] V. Chouhan et al., A Novel Transfer Learning Based Approach for Pneumonia Detection in Chest X-ray Images, Applied Sciences, 10(2), 559, 2020. doi:10.3390/app10020559.','[2] J. P. Cohen et al., TorchXRayVision, MIDL, 2022. https://mlmed.org/torchxrayvision/models.html','[3] H. Touvron et al., Training data-efficient image transformers & distillation through attention, ICML, 2021. https://github.com/facebookresearch/deit','[4] Abdullah et al., HybridCATNet, Frontiers in Digital Health, 2026. doi:10.3389/fdgth.2026.1769444.','[5] Kermany et al., Pediatric chest X-ray dataset, Mendeley Data v2. doi:10.17632/rscbjbr9sj.2. Data mirror: huggingface.co/datasets/hf-vision/chest-xray-pneumonia.'] :para(t)
doc=BaseDocTemplate(str(O/'Final_M4_M5_Paper.pdf'),pagesize=(612,792),leftMargin=36,rightMargin=36,topMargin=36,bottomMargin=36)
def paper_cover(c,doc):
 c.setFont('Times-Bold',24)
 for j,line in enumerate(['Lung-Focused Inputs for CNN-ViT','Pneumonia Classification']):c.drawCentredString(306,758-28*j,line)
 c.setFont('Times-Roman',10);c.drawCentredString(306,706,'Samarth Shekhar, Meghavi Rao, Rachit Kumar and Harshal Nerpagar')
 c.setFont('Times-Roman',8);c.drawCentredString(306,691,'Computer Vision Project / Final Milestones')
doc.addPageTemplates([PageTemplate(id='First',frames=[Frame(36,36,261,642,id='left'),Frame(315,36,261,642,id='right')],onPage=paper_cover,autoNextPageTemplate='TwoColumn'),PageTemplate(id='TwoColumn',frames=[Frame(36,36,261,720,id='left'),Frame(315,36,261,720,id='right')])])
doc.build(body)
(O/'TEAM_FINAL_PREP.txt').write_text('\n\n'.join(f'{i+1}. {title}\n'+ '\n'.join(lines) for i,(title,lines) in enumerate(slides))+'\n\nDefence: Never describe the frozen-feature pilot as a full late-block replication. Compare A/B/C/D within their own protocol. Explain negative results and resource limits openly.\n')
with (O/'TEAM_FINAL_PREP.txt').open('a') as f:
 f.write('''

QUESTIONS TO PREPARE

Sensitivity is the share of pneumonia images correctly detected. Specificity is the share of normal images correctly recognized. Macro F1 gives the normal and pneumonia class F1 scores equal weight. AUC measures ranking across thresholds, whereas the confusion matrix uses the declared decision threshold.

Why not compare M4 directly to M2 as proof of cropping? M2 trained late layers on the larger paper-training partition. M4 trains heads on frozen encoders using a separate validation split. Several settings change, so only matched A/B/C/D comparisons isolate the declared effects.

Is the crop only lung pixels? No. A bounding rectangle around both predicted lung masks, with a margin, still includes the heart and sometimes side markers. The masks have no manual ground truth here.

Did we train the segmenter? No. It is fixed pretrained PSPNet from TorchXRayVision. We use 256-pixel segmentation instead of its standard 512-pixel wrapper because the measured CPU runtime was about 4.4 times faster. The predicted box is mapped to original pixels before cropping.

Did we train a new transformer from scratch? No. DeiT-Tiny is pretrained on ImageNet. We train a small classifier head and select blend weights on validation data. Weight0 means the transformer contributes nothing, and weight1 means the transformer alone supplies that condition.

Does this prove clinical performance? No. This is a pediatric benchmark with a previously inspected test set, imperfect filename patient groups and a single experiment seed. External validation remains necessary.

''')
 f.write('Validation/test reversal: validation macro F1 '+str(val)+'. Cropping lowered validation performance but raised test performance. D was predeclared, not selected across conditions using test scores.\n')
 f.write('Measured crop verdict: '+verdict+'\n')
 f.write('B minus A macro F1: '+format(delta,'+.4f')+'; specificity: '+format(specificity,'+.2f')+' percentage points. Paired interval: '+ci_text+'\n')
 f.write('Validation-selected transformer weights: C='+str(conditions['C'].get('vit_weight'))+', D='+str(conditions['D'].get('vit_weight'))+'\n')
print('Built',O)
