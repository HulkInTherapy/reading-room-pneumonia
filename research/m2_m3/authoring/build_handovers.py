from pathlib import Path
import runpy
# The user's shared-content request supersedes the earlier assigned-slide guides.
runpy.run_path(str(Path(__file__).with_name('build_shared_guide.py')),run_name='__main__')
raise SystemExit(0)
import json,re
ROOT=Path('/Users/zebra/cvproj/m2_m3')
r=json.loads((ROOT/'results/summary.json').read_text());m=r['majority_vote']
fresh=False
f=ROOT/'experiments/restart15/results/independent_verification.json'
if f.exists() and json.loads(f.read_text()).get('verified'):
 fresh=True;m=json.loads((f.parent/'summary.json').read_text())['metrics']
pct=lambda x:f'{100*x:.2f}%'
common=f'''## Everything you need to know

Our project classifies children's chest X-rays into NORMAL or PNEUMONIA. Computer vision turns pixels into features, then a classifier predicts the label. We are evaluating a student research baseline, not making a clinical diagnosis tool.

M2 means reproduce a baseline that runs and show real results in a notebook. M3 means propose ONE change, explain why it might help, predict a numerical effect before implementing it, and add an introduction to the report and deck. We have not run the proposed change. M4 will test it after this hypothesis is submitted.

We chose **P7, Chouhan et al.2020**, replacing the tentative Kermany base from M1. Its five CNNs are AlexNet, DenseNet121, InceptionV3, ResNet18 and GoogLeNet. Their early layers stay frozen, but the paper trains some late convolutional blocks and classifiers. We reconstructed those training boundaries and the512-unit AlexNet head from the figures. Three pneumonia votes give the ensemble label. Our fixed five epochs are explicitly shorter than the paper schedule. Earlier head-only probes are archived separately, not used as the final ensemble.

P7 uses 5,232 training records, formed by merging the original 5,216 train and 16 validation records. The original 624 test records are kept separate. We found no exact-image hashes crossing train/test, but test includes six repeated records. No model was selected by its test score. The Mac ran the five models sequentially with two CPU threads and small batches.

**Our measured ensemble:** accuracy {pct(m['accuracy'])}, pneumonia recall {pct(m['sensitivity'])}, normal recall {pct(m['specificity'])}, macro F1 {m['macro_f1']:.3f}, AUC {m['auc']:.3f}. Confusion matrix: {m['confusion_matrix'][0][0]} normal images correctly normal, {m['confusion_matrix'][0][1]} normal images wrongly pneumonia, {m['confusion_matrix'][1][0]} pneumonia images missed, {m['confusion_matrix'][1][1]} pneumonia images correctly pneumonia.

The paper reports96.39% accuracy,99.62% recall and99.34% AUC. Our accuracy gap is {(m['accuracy']-.9639)*100:+.2f} percentage points. We reconstructed it independently because no author repository was verified. Our shortened five-epoch budget, fixed cached augmentation, float16 frozen-prefix caches and missing author settings can affect the result. It is a bounded reconstruction, not the complete paper schedule. A gap does not prove the authors are wrong.

Sensitivity means the fraction of pneumonia cases found. Specificity means the fraction of normal cases recognized. Precision asks how many pneumonia predictions were right. F1 balances precision and recall. Macro F1 averages the normal F1 and pneumonia F1 so both classes matter. AUC measures how well scores rank pneumonia above normal over possible thresholds. A high AUC can coexist with an unsuitable fixed threshold.

Always saying pneumonia gives 62.5% test accuracy and pneumonia F1 about 0.769, but **zero normal recall** and macro F1 about 0.385. So high pneumonia F1 alone does not mean balanced performance. Small data and imbalance do not automatically imply low F1 or overfitting.

**M3 one change:** use a lung-focused crop from a fixed pretrained segmentation model, with a 10% margin, then resize it to the same 224-pixel input. Keep everything else identical in paired future runs. The candidate is TorchXRayVision's pretrained ChestX-Det PSPNet, which has left/right lung channels. We verified its public documentation; we have not run its weights here. A crop may allocate more resolution to the lungs and reduce peripheral variation. It may also remove useful anatomy or produce bad pediatric masks.

**Prediction, not result:** macro F1 +0.01 to +0.03, normal recall +2 to +5 percentage points, AUC 0 to +0.02. Pneumonia recall may move -2 to +2 points and accuracy -1 to +2 points. These are relative to a matched full-image control under the future M4 protocol, not promises of new absolute scores.

The crop must use original-resolution source pixels before the final224-pixel resize. Our current Mac cache is already224 pixels, so cropping that cache cannot recover lost detail. M4 will fetch originals one shard at a time and use identical source pixels for control and crop.

Our live demo accepts a sample or an upload and computes the five trained CNN outputs. The displayed heatmap is real Grad-CAM from ResNet18 for the displayed class, not lung segmentation or an explanation of all five models. It starts from a coarse7-by-7 activation map. Image-left/right activation shares do not establish patient anatomy. The crop proposal remains untested.

M4 will make a proper group-aware validation split, rerun a full-image control on that split, and compare cropping with identical settings. We must not compare a crop trained on a different split directly to today's M2 score. Paired seeds and bootstrap differences will measure uncertainty. A failure to improve is still useful research when honestly analyzed.

**Corrections from M1:** P9/Singh 2024 already reports F1 0.952, AUC 0.966 and confidence intervals. It uses a different split and a large DeiT-Base transformer. Do not say nobody has reported F1, all later papers use the exact same test set, every improvement is noise, or CNN-ViT hybrids are untried. Do not promise state of the art.

The three foundations: Kermany provided the pediatric dataset and transfer-learning example; CheXNet demonstrated DenseNet-based chest X-ray classification on a different dataset; Wang's ChestX-ray8 gave large-scale chest X-ray data and localization benchmarks. They are context, not our reproduced model.

'''
scripts={
'SAMARTH':('Slides1,4 and9','''Our task is normal-versus-pneumonia classification on pediatric X-rays. We selected Chouhan's five-CNN ensemble and reconstructed its frozen early layers, trainable late blocks and voting rule. Our local five epochs are explicitly shorter than the paper schedule. This replaces the tentative M1 baseline while retaining lung cropping. We merged train and validation to match5232 training records and kept the original624 test records separate. The actual learned models also drive our upload-and-heatmap demo. M3 proposes cropping as one variable; its predicted improvement has not been tested. We will submit it before implementation and report the result even if it fails.'''),
'MEGHAVI':('Slides 2 and 5',f'''The project first establishes a working published baseline, then predicts the effect of one change. In the baseline, pretrained CNN features feed trained classifiers and five models vote. Our actual ensemble accuracy is {pct(m['accuracy'])}, pneumonia sensitivity is {pct(m['sensitivity'])}, normal specificity is {pct(m['specificity'])}, macro F1 is {m['macro_f1']:.3f}, and AUC is {m['auc']:.3f}. These values come from real saved per-image outputs. Sensitivity measures pneumonia detection; specificity measures normal recognition. Macro F1 gives both classes equal importance. We therefore report more than accuracy. This is a single-hospital benchmark result, with no clinical deployment claim.'''),
'RACHIT':('Slides 3 and 7','''Our M1 literature review compared compact CNNs, pretrained backbones, a five-CNN ensemble and a transformer. We chose the ensemble for reconstruction, not because it is the newest model, but because it gives us a runnable reference. P9 uses DeiT-Base and a different split, and already reports F1 and AUC. Our one proposed change is a lung-focused crop. A pretrained anatomical segmenter supplies left and right lung masks. Their bounding box, with a fixed margin, is resized to the same input size. This may devote more pixels to lung tissue, but mask errors can hurt. We will hold the classifiers, vote, schedule and split constant so the comparison isolates that input change.'''),
'HARSHAL':('Slides 6 and 8',f'''The confusion matrix shows what our headline score hides. We got {m['confusion_matrix'][0][1]} false pneumonia predictions among normal images and missed {m['confusion_matrix'][1][0]} pneumonia images. The paper accuracy is 96.39%, and our gap is {(m['accuracy']-.9639)*100:+.2f} percentage points. We disclose the gap and reconstruction settings rather than calling the paper wrong. For M3 we predict cropping will raise macro F1 by one to three hundredths and normal recall by two to five percentage points. These are predictions, not measured improvements. We will test them with an identically trained full-image control under the new validation protocol. If the result is flat or worse, the hypothesis is not supported and we will report that.'''),
}
proposal='''**M3 proposed final system:** crop both lungs with10% margin from original-resolution images, then run the five CNNs plus an ImageNet-pretrained small ViT such as DeiT-Tiny. Combine probability scores using validation-selected weights. This is not an exact P9 reproduction. We have not trained crops or a ViT yet.

Cropping may devote more pixels to lung tissue and reduce peripheral distractions. The ViT uses self-attention between image patches and may correct different errors from the CNNs. Neither guarantees a gain or creates additional independent clinical data. Bad crop masks can hurt; five highly correlated CNNs plus an equally biased ViT may still fail together.

There are two interventions. Cropping remains the primary hypothesis for the handout's one-change requirement, and ViT is an explicit second extension. To test our combined target, run A full-image CNNs, B cropped CNNs, C full-image CNN+ViT, D cropped CNN+ViT. B-A tests crop, C-A tests ViT, D-B ViT on crops, D-C crop with the hybrid. D is the target system.

**Predictions only:** cropping B-A macro F1 +0.01 to +0.03 and specificity +2 to +5 percentage points. ViT under matched aggregation macro F1 +0.01 to +0.03 and accuracy +1 to +3 points. Combined D-A macro F1 +0.01 to +0.04 and accuracy0 to +3 points, with pneumonia recall no more than2 points lower. Do not add the two gains together. Improvement can be zero or negative, and cannot exceed metric ceilings.

Soft voting means combining probability scores. A six-model hard vote can tie. Our proposed score gives the five-CNN mean weight1-w and the ViT weightw, chosen from a small predeclared grid on validation only. CNN-only comparisons use the same aggregation and threshold-selection process withw=0. We separately report P7's original majority vote, so changing aggregation cannot masquerade as a transformer benefit. Never tune on test.

The segmenter candidate is TorchXRayVision ChestX-Det PSPNet with left/right lung channels. Crop before224 resizing; enlarging our already224-pixel cache cannot restore detail. Redownload originals one shard at a time. A failed crop falls back to the whole image and its failure is counted. Start ViT with a frozen-backbone classifier pilot to check Mac memory/time before limited fine-tuning.

JEPA and mixup are optional later work. I-JEPA learns by predicting representations of hidden image regions, rather than pixels; its pretrained encoders learned from large datasets. V-JEPA is mainly video/world modelling. Mixup blends pairs of training images and labels; it is not LeCun's JEPA approach. None is implemented in this submission.

'''
common=re.sub(r'\*\*M3 one change:\*\*.*?Our live demo',proposal+'Our live demo',common,flags=re.S)
common=common.replace('M4 will make a proper group-aware validation split, rerun a full-image control on that split, and compare cropping with identical settings.','M4 will use a fixed group-aware validation protocol and run the four matched crop/ViT comparisons with comparable settings.')
common=common.replace('The crop proposal remains untested.','The crop and ViT proposal remains untested.')
if fresh:
 common=common.replace('Our fixed five epochs are explicitly shorter than the paper schedule.','We completed15 epochs per model in the audited restart, still shorter than the paper\'s50 to200 epoch schedules. The original five-epoch run is preserved.')
 common=common.replace('Our shortened five-epoch budget, fixed cached augmentation, float16 frozen-prefix caches and missing author settings can affect the result.','The restart uses4164 train and1042 group-validation images, four cyclic cached views, batch16 and validation-selected checkpoints. Prefix caches and unspecified author settings remain departures. Multiple settings changed, so comparison to the old run does not isolate epochs. Test was previously inspected and is not a blind external holdout.')
 common=common.replace('The original 624 test records are kept separate.','Our restart instead reserves1042 of5206 unique non-test records for group validation and trains on4164. The original624 test records remain separate.')
scripts={
 'SAMARTH':('Slides1,4 and9','Our project classifies pediatric X-rays as normal or pneumonia. P7 supplies the five-CNN baseline. We report actual measured results and explicitly shortened training schedules. Our final proposed system adds both lung cropping and a pretrained small ViT, but neither improvement has run yet. The current demo performs real CNN inference and a ResNet18 component heatmap. We will test the combined proposal with separate controls rather than claim that both changes already helped.'),
 'MEGHAVI':('Slides2 and5',f'Our first objective is a baseline that runs. The measured ensemble accuracy is {pct(m["accuracy"])}, precision {pct(m["precision"])}, sensitivity {pct(m["sensitivity"])}, specificity {pct(m["specificity"])}, macro F1 {m["macro_f1"]:.3f} and AUC {m["auc"]:.3f}. Sensitivity measures pneumonia cases found; specificity measures normal cases recognized. For M3 we propose lung-focused input and complementary CNN and transformer predictions. The numerical improvements remain hypotheses. Validation performance and test performance must not be confused.'),
 'RACHIT':('Slides3 and7','P7 combines five CNNs. P9 uses a larger pretrained DeiT-Base transformer and a different split; it already reports F1 and AUC. Our proposal uses a smaller pretrained ViT candidate to fit the Mac, not an exact P9 reproduction. First crop the union of both lungs with10% margin, then feed that region to both CNNs and ViT. CNNs learn local patterns and self-attention links image patches, so their errors may complement each other. We choose probability weights using validation, not test. Cropping is the primary handout hypothesis and the transformer is an explicit second extension.'),
 'HARSHAL':('Slides6 and8',f'Our actual confusion matrix contains {m["confusion_matrix"][0][1]} false positives and {m["confusion_matrix"][1][0]} missed pneumonia cases. For the new proposal we need four controls: A full-image CNNs, B cropped CNNs, C full-image CNN plus ViT and D cropped CNN plus ViT. D is our final target. Our combined prediction relative to A is macro F1 plus0.01 to0.04 and accuracy zero to three percentage points, with recall loss no more than two points. These are predictions, not results. Matching probability aggregation matters because changing the voting rule alone could change scores. No improvement is a valid outcome and must be reported honestly.')
}
questions='''## Questions you can answer without looking anything up

**Why P7 rather than the newer transformer?** The assignment is research reproduction. P7 is practical and has an explicit ensemble rule. P9 needs a larger fine-tuning run and its split is not recoverable exactly. Newer does not automatically mean more reproducible.

**Did we reproduce the exact author's code and full schedule?** No. We reconstructed the trainable architecture boundaries using the paper figures and official backbones. Our measured five epochs are shorter than its full schedule. We disclose this, other assumptions and the reproduction gap. The corrected baseline actually runs.

**Did we prove the paper is wrong?** No. We measured our reconstruction and explained the gap. Different implementation choices can produce different scores.

**Did we improve the model yet?** No. M3 asks for a hypothesis before implementing the change. The crop improvement is still a prediction.

**Why can a crop help?** It changes how input pixels are allocated and reduces peripheral variation. This is a testable mechanism, not a guarantee. Poor masks or missing anatomy may hurt.

**Are we adding both crop and transformer?** Yes, as the final proposed system. They are two changes, so A/B/C/D comparisons isolate their individual and combined effects. Existing hybrids mean we cannot claim global novelty. Neither improvement has been trained yet.

**What does the demo heatmap show?** Grad-CAM uses actual gradients and ResNet18 activation maps for the displayed class. It is a component explanation, not a lesion boundary or the explanation of all five models. The demo has not implemented the lung-crop proposal.

**Why is the validation set merged?** To reconstruct P7's stated partition. That leaves no held-out validation subset in the M2 comparison, so budgets and thresholds were fixed. M4 will establish a proper validation protocol and rerun both conditions under it.

**What if the teacher wants a result we did not measure?** Say: “We have not measured that yet. It is a limitation or a planned later experiment.” Do not guess a number.

## Files to submit

M2_Reproduced_Baseline.ipynb contains code and executed verification/results. Reading_Room_M2_M3.pptx contains the introduction, baseline and hypothesis. M2_M3_Project_Report.docx contains the introduction, literature summary, actual results and proposal. The PDF deck is for easy viewing. The per-image CSV and source code support reproducibility.
'''
for name,(pages,script) in scripts.items():
 if fresh:
  questions=questions.replace('Our measured five epochs are shorter than its full schedule.','Our original five epochs and audited15-epoch restart are both shorter than its full schedule.')
  questions=questions.replace('The crop improvement is still a prediction.','The crop and transformer improvements are still predictions; the15-epoch run is baseline formation.')
  questions=questions.replace('M4 will establish a proper validation protocol and rerun both conditions under it.','The audited restart already establishes group validation; M4 will reuse a declared matched protocol for all four conditions.')
  questions=questions.replace('M2_Reproduced_Baseline.ipynb contains code and executed verification/results.','M2_Audited_15_Epoch_Restart.ipynb contains the latest executed verification/results. M2_Reproduced_Baseline.ipynb preserves the original five-epoch experiment.')
  questions+='\n\n**Did15 epochs improve the test score?** No. Accuracy moved from87.18% to85.74%, precision from83.12% to81.55%, and false positives from79 to88. Recall remained99.74%. Selected validation accuracy was98.94%. This gap does not identify one cause. Fewer training images and multiple setting changes prevent blaming epochs alone. Do not hide this negative result or claim crops/ViT will certainly fix it.\n'
 (ROOT/'output'/f'{name}_M2_M3_PREP.md').write_text(f'# {name.title()} preparation for M2 and M3\n\nYour speaking sections: {pages}. Read the context first, then rehearse the script.\n\n'+common+'## Your speaking script\n\n'+script+'\n\n'+questions)
print('Four personal guides created')
