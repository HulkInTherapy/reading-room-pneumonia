"""Report uses real measured artifacts only; run with bundled artifact Python."""
import json,re
from pathlib import Path
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH,WD_BREAK
ROOT=Path('/Users/zebra/cvproj/m2_m3')
s=json.loads((ROOT/'results/summary.json').read_text());m=s['majority_vote']
fresh=False;model_metrics={n:json.loads((ROOT/'results'/f'{n}.json').read_text())['metrics'] for n in s['models']}
f=ROOT/'experiments/restart15/results/independent_verification.json'
if f.exists() and (v:=json.loads(f.read_text())).get('verified'):
 fresh=True;restart=json.loads((f.parent/'summary.json').read_text());m=restart['metrics'];model_metrics={r['model']:r['test_metrics'] for r in v['models']}
online=None
f=ROOT/'experiments/paper_online5/results/independent_verification.json'
if f.exists() and json.loads(f.read_text()).get('verified'):
 online=json.loads((f.parent/'summary.json').read_text());m=online['metrics'];model_metrics=online['model_metrics']
d=Document();sec=d.sections[0];sec.top_margin=sec.bottom_margin=Inches(.65);sec.left_margin=sec.right_margin=Inches(.7)
for name in ['Normal','Title','Heading 1','Heading 2']:
 st=d.styles[name];st.font.name='Arial';st.font.color.rgb=RGBColor(0,0,0)
d.styles['Normal'].font.size=Pt(10.5);d.styles['Normal'].paragraph_format.space_after=Pt(7);d.styles['Normal'].paragraph_format.line_spacing=1.08
d.styles['Title'].font.size=Pt(21);d.styles['Heading 1'].font.size=Pt(15);d.styles['Heading 2'].font.size=Pt(12)
for node in list(d.styles.element.xpath('//w:pBdr')):
 node.getparent().remove(node)
def readable(x):
 x=re.sub(r"\b(with|by|than|using|and|Table|Figures|batch|before|among|at|accuracy|reports|to|from|uses|completed|original|test|audited|of|seed|seeds|on|paper's|Our|stated|has|precision|was|all|cached|obtain|remains|the|reserved|trained|every|after|LR|audit)(?=\d)",r'\1 ',str(x))
 x=re.sub(r'(?<=[A-Za-z]),(?=\d)',', ',x)
 return re.sub(r'(?<=\d)(?=(?:training|group|epochs|epoch|points|records|images|iterations|GiB|pixel)\b)',' ',x)
def para(x):d.add_paragraph(readable(x))
next_heading_new_page=False
def heading(x):
 global next_heading_new_page
 p=d.add_heading(x,level=1)
 if next_heading_new_page:p.paragraph_format.page_break_before=True;next_heading_new_page=False
def pagebreak():
 global next_heading_new_page
 next_heading_new_page=True
def table(rows,widths=None,size=9.5):
 t=d.add_table(rows=0,cols=len(rows[0]));t.autofit=False
 if widths:
  for col,width in zip(t.columns,widths):col.width=Inches(width)
 for i,row in enumerate(rows):
  cells=t.add_row().cells
  for j,txt in enumerate(row):
   cells[j].text=readable(txt)
   cells[j].vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
   if widths:cells[j].width=Inches(widths[j])
   tcPr=cells[j]._tc.get_or_add_tcPr();borders=OxmlElement('w:tcBorders')
   for edge in ['top','left','bottom','right']:
    e=OxmlElement('w:'+edge);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');borders.append(e)
   tcPr.append(borders)
   sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'DCEBEE' if i==0 else ('F5F7F8' if i%2==0 else 'FFFFFF'));tcPr.append(sh)
   margins=OxmlElement('w:tcMar')
   for edge in ['top','left','bottom','right']:
    e=OxmlElement('w:'+edge);e.set(qn('w:w'),'80');e.set(qn('w:type'),'dxa');margins.append(e)
   tcPr.append(margins)
   for p in cells[j].paragraphs:
    if j>0 and re.fullmatch(r'[-+]?\d+(\.\d+)?%?',str(txt)):p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after=Pt(3);p.paragraph_format.space_before=Pt(3)
    for run in p.runs:run.font.size=Pt(size);run.bold=(i==0)
 spacer=d.add_paragraph();spacer.paragraph_format.space_after=Pt(3);spacer.paragraph_format.space_before=Pt(0)
 spacer.paragraph_format.line_spacing=Pt(2)
 return t
def pct(x):return f'{100*x:.2f}%'
d.add_paragraph('Pneumonia Classification Baseline and Proposed Crop and Transformer System',style='Title')
para('Milestones 2 and 3   6 October 2026')
para('Samarth Shekhar 230042, Meghavi Rao 230044, Rachit Kumar 230128, Harshal Nerpagar 230076')
heading('Introduction')
intro=(ROOT/'INTRODUCTION.md').read_text().split('## References')[0]
if online:intro=intro.replace('We train its stated late blocks and classifiers for five epochs, shorter than the paper\'s full schedule, and report that limitation alongside the reproduction gap.','Our latest reconstruction uses five fixed epochs on all5232 paper-training records, full FP32 image forwards and fresh augmentation. Frozen early weights remain fixed while BatchNorm statistics adapt. Three models trained on the Mac and two on an isolated CPU VPS. We preserve earlier experiments and disclose that this shorter budget and unspecified author settings prevent exact full-schedule replication.')
elif fresh:intro=intro.replace('We train its stated late blocks and classifiers for five epochs, shorter than the paper\'s full schedule, and report that limitation alongside the reproduction gap.','We preserve the initial five-epoch reconstruction and report the audited15-epoch restart, using4164 training and1042 group-validation images, with checkpoints selected by validation macro F1. Fifteen epochs remains shorter than the paper\'s50 to200 epoch schedules. Several settings changed together, so this comparison does not isolate training duration.')
for p in intro.split('\n\n'):
 if not p.startswith('#'):para(p.strip())
pagebreak();heading('Literature Survey Summary')
para('The M1 survey examined the following methods and their relevance to the proposed baseline. Different datasets and split protocols prevent a direct ranking from reported accuracy alone. We select P7, replacing the tentative Kermany baseline while retaining the M1 lung-cropping proposal.')
table([
 ['Paper','Method','Role in this project'],
 ['Kermany et al 2018','Image-based transfer learning and pediatric CXR data','Dataset foundation'],
 ['Rajpurkar et al 2017','CheXNet DenseNet121 on ChestX-ray14','CNN transfer-learning context'],
 ['Wang et al 2017','ChestX-ray8 dataset and localization benchmarks','Dataset and evaluation context'],
 ['Stephen et al 2019','CNN trained from scratch','Alternative to pretrained features'],
 ['Ayan and Unver 2019','VGG16 and Xception comparison','Backbone comparison'],
 ['Rahman et al 2020','Multiple pretrained CNN backbones','Transfer-learning comparison'],
 ['Chouhan et al 2020','Five CNNs with frozen early and trainable late layers','Selected base paper'],
 ['Rajaraman et al 2018','Customized CNN and activation-map inspection','Qualitative interpretation'],
 ['Singh et al 2024','Pretrained DeiT-Base Vision Transformer','Recent alternative with different split'],
],[1.65,2.5,2.5],9.5)
heading('Updated Reading of the Transformer Paper')
para('Singh et al. report F score 0.952 and AUC 0.966 in Table 5, and a confidence-interval section. Its stated split is 4684/586/586, not the original 624-record test set. Table 3 class totals contradict Table 2. These facts correct the M1 claims that every paper omitted F1 or uncertainty and used the identical test images. Neither imbalance nor small dataset size proves overfitting or invalidity.')
heading('Research Focus')
para('Our contribution is a documented reconstruction followed by a crop and transformer proposal with separate controls. Cropping remains the primary M3 change and the ViT is an explicit second extension. We do not claim that either technique or their combination is globally new. Hybrid transformer ensemble approaches have already been published [4]. Four matched comparisons will test their individual and combined effects.')
para('The M1 review also included Zech et al. on cross-hospital generalization, DeGrave et al. on shortcut learning in COVID radiographs, and Saporta et al. on saliency evaluation against expert annotations. These provide evaluation context and are not the selected baseline or evidence that our own model uses shortcuts.')
pagebreak();heading('Baseline Reconstruction and Measured Results')
para('The selected paper is Chouhan et al. [2]. We reconstruct Figures4-8: train ResNet layer4, DenseNet block4 layers15/16, Inception Mixed_7c and GoogLeNet inception5b plus classifiers, freezing earlier weights. AlexNet uses the stated9216->512->2 replacement head. Three pneumonia votes give the label. '+('The final online run completed five epochs per model on5232 training records, with fixed final checkpoints and no independent validation subset. Test remains624 records. Earlier experiments used cached features; the15-epoch audit reserved1042 validation images and trained4164. Several changes are bundled, so comparison does not isolate epochs.' if online else 'The audited restart completed15 epochs per model, using4164 training and1042 group-validation images. Validation selected checkpoints before evaluation on the original624 test records. The previously inspected test set is not a blind external holdout.' if fresh else 'The original completed baseline uses five epochs,5232 training and624 test records.'))
table([['Model','Accuracy','Sensitivity','Specificity','Macro F1','AUC']]+[
 [n,pct((r:=model_metrics[n])['accuracy']),pct(r['sensitivity']),pct(r['specificity']),f"{r['macro_f1']:.3f}",f"{r['auc']:.3f}"] for n in s['models']]+[
 ['Ensemble',pct(m['accuracy']),pct(m['sensitivity']),pct(m['specificity']),f"{m['macro_f1']:.3f}",f"{m['auc']:.3f}"]],None,9.1)
para(f"The paper reports 96.39% accuracy, 99.62% sensitivity, 93.28% precision and 99.34% AUC. Our accuracy difference is {(m['accuracy']-.9639)*100:+.2f} percentage points, and our precision is {pct(m['precision'])}. Pneumonia F1 is {m['pneumonia_f1']:.3f} and normal F1 is {m['normal_f1']:.3f}. ROC AUC uses mean probabilities, while labels use majority voting.")
table([['Actual and predicted','Predicted normal','Predicted pneumonia'],['Actual normal',*m['confusion_matrix'][0]],['Actual pneumonia',*m['confusion_matrix'][1]]],None,10)
heading('Reconstruction Assumptions')
para('No author training repository was verified. Paper Table2 reports50 to200 epochs. '+('Our final run uses five epochs, batch16, fresh per-example augmentation, FP32 full-image forwards and updating BatchNorm statistics. Adam uses LR0.001 halved every3 epochs; AlexNet switches after200 steps to0.00001. Exact augmentation/noise and BatchNorm settings remain local assumptions. ' if online else 'Our15-epoch restart uses batch16, four cyclic views, assumed noise and validation selection. Float16 prefix caches convert to float32 for training; early BatchNorm statistics stay frozen. ')+'The paper metrics do not reconcile to one stated624-image confusion matrix, possibly because of averaging or another definition. We cannot claim exact replication.')
heading('Verification and Limits')
para('Source shards passed SHA256 checks;624 test records contain618 unique images. '+('Full checkpoint reloads reproduced every test probability. Epoch, training-record and BatchNorm counts passed verification. The corrected run improves accuracy/precision but reduces pneumonia recall and AUC relative to the cached audit; bundled changes prevent causal attribution. ' if online else 'The15-epoch audit had98.94% validation accuracy but85.74% test accuracy. ')+'The previously inspected benchmark is not blind and patient identifiers remain proxies. One seed does not establish generalization. The demo retains original weights and component Grad-CAM, not lesion localization.')
pagebreak();heading('Hypothesis and Numerical Prediction')
para('Our final target combines lung cropping and a pretrained small Vision Transformer with the five CNNs. Cropping remains the primary M3 hypothesis required by the handout. Adding the transformer is an explicit second extension, not a one-variable experiment. Neither improvement has been implemented or measured.')
para('Use a fixed pretrained lung segmenter, crop both lungs with10% margin from original-resolution pixels, then resize to224. Failed masks fall back to the full image and are counted. Cropping may allocate more resolution to lung tissue but may remove useful anatomy. An ImageNet-pretrained DeiT-Tiny candidate adds patch self-attention and may contribute complementary errors. It is a bounded candidate rather than P9\'s exact DeiT-Base reproduction.')
table([['Run','Image region','Classifiers'],['A','Full X-ray','Five CNNs'],['B','Lung crop','Five CNNs'],['C','Full X-ray','CNNs plus pretrained ViT'],['D','Lung crop','CNNs plus pretrained ViT']],None,10)
para('B-A isolates cropping, C-A adding ViT, D-B adding ViT on crops and D-C cropping in the hybrid. All four use matching probability aggregation. Original P7 majority voting remains a separate reported baseline.')
table([['Comparison','Numerical prediction before implementation'],['Crop B-A','Macro F1 +0.01 to +0.03; specificity +2 to +5 points'],['ViT C-A or D-B','Macro F1 +0.01 to +0.03; accuracy +1 to +3 points'],['Combined D-A','Macro F1 +0.01 to +0.04; accuracy0 to +3 points'],['Recall safeguard','Pneumonia recall loss no more than2 points']],None,10)
heading('Planned Experiment and Limitations')
para('After submitting the proposal, use the same group split, source pixels, per-model training schedules and augmentation in the four conditions. Begin the ViT with a frozen-backbone classifier pilot. The hybrid score is a weighted mean of CNN and ViT probabilities. Select checkpoints, a small predeclared grid of weights and thresholds on validation only, subject to the recall safeguard. Match that threshold procedure for CNN-only aggregation. Never use test scores to choose settings. Run paired seeds42,43,44 if resources permit, report paired uncertainty and record crop failures and runtime.')
para('These gains are predictions and need not add together. Metric ceilings limit improvement. No macro-F1 gain or recall loss beyond2 points fails to support the combined proposal. Single-center pediatric data and imperfect segmentation limit conclusions. JEPA feature transfer and mixup remain optional later work, outside these primary comparisons. The demo does not yet implement cropping or ViT. We make no clinical, global novelty or state-of-the-art claim.')
pagebreak();heading('References')
for ref in [
 '[1] D. Kermany, K. Zhang and M. Goldbaum, Mendeley Data V2, 2018. doi:10.17632/rscbjbr9sj.2.',
 '[2] V. Chouhan et al., A Novel Transfer Learning Based Approach for Pneumonia Detection in Chest X-ray Images, Applied Sciences 10(2), 559, 2020. doi:10.3390/app10020559.',
 '[3] S. Singh et al., Efficient pneumonia detection using Vision Transformers on chest X-rays, Scientific Reports 14, 2487, 2024. doi:10.1038/s41598-024-52703-2.',
 '[4] A hybrid explainable ensemble transformer encoder for pneumonia identification from chest X-ray images, 2023. https://pmc.ncbi.nlm.nih.gov/articles/PMC10248870/.',
 '[5] D.S. Kermany et al., Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning, Cell 172, 2018.',
 '[6] P. Rajpurkar et al., CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning, 2017. https://arxiv.org/abs/1711.05225.',
 '[7] X. Wang et al., ChestX-ray8: Hospital-Scale Chest X-Ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases, CVPR 2017.',
 '[8] O. Stephen et al., An Efficient Deep Learning Approach to Pneumonia Classification in Healthcare, 2019. doi:10.1155/2019/4180949.',
 '[9] E. Ayan and H.M. Unver, Diagnosis of Pneumonia from Chest X-Ray Images Using Deep Learning, IEEE EBBT 2019.',
 '[10] T. Rahman et al., Transfer Learning with Deep Convolutional Neural Network (CNN) for Pneumonia Detection using Chest X-ray, Applied Sciences 10(9), 3233, 2020. doi:10.3390/app10093233.',
 '[11] S. Rajaraman et al., Visualization and Interpretation of Convolutional Neural Network Predictions in Detecting Pneumonia in Pediatric Chest Radiographs, 2018. doi:10.3390/app8101715.',
 '[12] J.R. Zech et al., Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs, 2018. https://pmc.ncbi.nlm.nih.gov/articles/PMC6219764/.',
 '[13] A.J. DeGrave et al., AI for radiographic COVID-19 detection selects shortcuts over signal, 2021. doi:10.1038/s42256-021-00338-7.',
 '[14] A. Saporta et al., Benchmarking saliency methods for chest X-ray interpretation, 2022. doi:10.1038/s42256-022-00536-x.',
]:
 p=d.add_paragraph(ref);p.paragraph_format.space_after=Pt(5)
 for r in p.runs:r.font.size=Pt(9.5)
for node in list(d.element.xpath('//w:pBdr')):
 node.getparent().remove(node)
d.save(ROOT/'output/M2_M3_Project_Report.docx')
print('Report created')
