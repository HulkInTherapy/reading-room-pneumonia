"""One common plain-language guide aligned with the nine-slide deck."""
from pathlib import Path
import json,datetime,hashlib
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output'
initial=json.loads((ROOT/'results/summary.json').read_text())['majority_vote']
audit=json.loads((ROOT/'experiments/restart15/results/summary.json').read_text())['metrics']
latest=None
receipt=ROOT/'experiments/paper_online5/results/independent_verification.json'
if receipt.exists() and json.loads(receipt.read_text()).get('verified'):
    latest=json.loads((receipt.parent/'summary.json').read_text())['metrics']
def pct(x):return f'{100*x:.2f}%'
pending='The corrected five-epoch run is still in progress. It has no final verified test result yet. The pending cells on the deck mean exactly that; do not substitute its training accuracy or guess a result.'
if latest:
    pending=f'The corrected five-epoch run finished and passed independent verification. Its test accuracy is {pct(latest["accuracy"])}, precision {pct(latest["precision"])}, pneumonia recall {pct(latest["sensitivity"])}, normal recall {pct(latest["specificity"])}, macro F1 {latest["macro_f1"]:.3f} and AUC {latest["auc"]:.3f}. These are measured results, not predictions.'
shown=latest or audit
tradeoff=''
if latest:
    tradeoff=f'The corrected run improved accuracy and precision relative to the cached audit, but not every metric improved. Pneumonia recall changed from {pct(audit["sensitivity"])} to {pct(latest["sensitivity"])} and AUC from {audit["auc"]:.3f} to {latest["auc"]:.3f}. It made {latest["confusion_matrix"][0][1]} false pneumonia predictions and missed {latest["confusion_matrix"][1][0]} pneumonia cases. We must report that trade-off; the bundled changes do not identify which correction caused the improvement.'
text=f'''# Shared preparation for the pneumonia project

For Samarth Shekhar, Meghavi Rao, Rachit Kumar and Harshal Nerpagar. Everyone gets the same explanation. There are no assigned speaking sections. This guide follows the content of our nine-slide Reading the Reading Room deck, so anyone can explain any part.

Updated {datetime.datetime.now().strftime('%d %B %Y at %H:%M IST')}. **{pending}**

## The project in simple words

We give a computer a child's chest X-ray and ask it to predict one of two dataset labels: normal or pneumonia. Computer vision means learning useful patterns from image pixels. The project studies how well a published method works on a public benchmark. It is not a tool that can safely diagnose a real patient.

M1 was our literature review: what other researchers did and which method to use. M2 means build a working baseline by reconstructing a selected paper and show real code and results. A baseline is the reference system we later try to improve. M3 means explain the proposed improvement, its mechanism and a numerical prediction before implementing it. M4 will implement the changes and compare them. M5 will analyze errors and write up the final work. Our evaluation now covers M2 and M3.

## The paper and why we chose it

Our exact base paper is **Chouhan et al., 2020, A Novel Transfer Learning Based Approach for Pneumonia Detection in Chest X-ray Images**. In our literature list, it is **P7**.

It combines five existing image models: AlexNet, DenseNet121, GoogLeNet, InceptionV3 and ResNet18. Each is a CNN, or convolutional neural network: a model that builds visual features from small image regions and combines them into more complex patterns. They start with ImageNet-pretrained weights. Transfer learning means reusing that earlier visual learning rather than learning everything from our small X-ray collection.

We keep the paper's early weights fixed and train its specified late blocks and final classifiers. The classifier produces normal and pneumonia scores. Each model votes for one class, and at least three pneumonia votes make the final label pneumonia. An ensemble is this combination of several models.

We chose P7 because its five-model structure and voting rule give us a practical reference to reconstruct. P9, Singh's 2024 Vision Transformer paper, is useful context but uses a larger DeiT-Base model and a different split. We have not verified author-released training repositories for either exact paper. We use official backbone implementations and disclose assumptions. A generic GitHub pneumonia project is not necessarily the paper's code.

## What the literature slide means

You do not need to explain every paper's internals. Know the role of each row:

- **P1 Kermany 2018:** the pediatric X-ray dataset and an image-based transfer-learning example.
- **P2 CheXNet 2017:** DenseNet-based chest X-ray classification on a different dataset.
- **P3 Wang 2017:** a larger chest X-ray dataset and localization benchmarks.
- **P4 Stephen 2019:** a compact CNN trained from scratch.
- **P5 Ayan and Unver 2019:** VGG16 versus Xception comparison.
- **P6 Rahman 2020:** comparison of pretrained CNN backbones.
- **P7 Chouhan 2020:** the five-CNN ensemble we selected.
- **P8 Rajaraman 2018:** classification with activation-map interpretation.
- **P9 Singh 2024:** a pretrained Vision Transformer approach.

The first three give background, not three models we reproduced. P9 already reports F1, AUC and confidence intervals. Different splits mean we cannot directly rank every paper by accuracy. CNN-transformer hybrids already exist, so our proposed combination is not a global first.

## The dataset and the split

The Kermany collection contains 5,216 original training records, only 16 original validation records, and 624 test records. It contains pediatric images from one hospital, with roughly three pneumonia-labelled training images for every normal one. These are limitations, not automatic proof that the labels or models are invalid.

Training images teach the model. Validation images help choose settings or checkpoints without using the test score. Test images measure the final chosen system. An epoch means one pass through all training records. A batch is the smaller group processed together.

P7 combines the original training and validation records: **5,232 training records, including 1,349 normal and 3,883 pneumonia**. Its test partition has **624 records, including 234 normal and 390 pneumonia**. This is the paper-partition table in our deck. Our second experiment used a different training/validation split, which we must explain rather than hide.

## The three runs and what changed

We have three main experiments. Small debugging probes are not additional headline baselines.

| Experiment | Epochs | Training records | Test accuracy | Test precision |
| --- | --- | --- | --- | --- |
| Initial cached run | 5 | 5,232 | {pct(initial['accuracy'])} | {pct(initial['precision'])} |
| Cached audit run | 15 | 4,164 | {pct(audit['accuracy'])} | {pct(audit['precision'])} |
| Corrected online run | 5 | 5,232 | {pct(latest['accuracy']) if latest else 'Pending'} | {pct(latest['precision']) if latest else 'Pending'} |

**First run:** to reduce computation, we saved early image features and reused them. We used a clean and one fixed augmented image version. Augmentation means slightly changing an image during training, such as flipping it or adjusting brightness. We trained the paper's late layers for five epochs. This was a resource-bounded reconstruction, not the paper's full schedule.

**Second run:** we removed repeated non-test records and reserved 1,042 images for group-based validation, leaving 4,164 for training. We trained every model for 15 epochs using four saved views, one view for an entire epoch. We selected each model using validation performance, not test. A checkpoint is a saved model state; the chosen epochs were 13, 9, 5, 13 and 1 for ResNet, DenseNet, GoogLeNet, Inception and AlexNet respectively. All completed 15 epochs, even when an earlier checkpoint was selected.

The selected ensemble had **98.94% validation accuracy but 85.74% test accuracy**. It correctly found 389 of 390 pneumonia cases, but wrongly called 88 of 234 normal cases pneumonia. Fifty-four normal images received pneumonia votes from all five models. More epochs did not solve their shared errors. The changed split and settings mean this was not an isolated test of epoch count.

**Third run:** we restored all 5,232 paper-training records. We removed saved-feature caching and use fresh image augmentation for every training pass. Complete image forwards use FP32, ordinary full-precision arithmetic. Early weights still stay fixed, while BatchNorm's internal feature averages and variances update during training. This removes the previous frozen-statistics approximation, but the authors' exact BatchNorm policy is unknown. We use the final epoch-five checkpoints, fixed in advance, rather than selecting using test scores. There is no independent validation subset in this full-paper-partition run.

The Mac runs ResNet, GoogLeNet and AlexNet sequentially; the isolated VPS runs DenseNet and Inception sequentially. Both use two CPU threads and matching library versions. Final evaluation and verification happen on the Mac. Parallel machines do not change the five-model voting method.

{tradeoff}

## How to explain the scores and the gap

- **Accuracy:** how many of all predictions were correct.
- **Precision:** among images called pneumonia, how many really had that dataset label.
- **Recall or sensitivity:** how many pneumonia-labelled images we found.
- **Specificity:** how many normal images we correctly recognized.
- **F1:** a balance of precision and recall. **Macro F1** gives normal and pneumonia equal importance by averaging their F1 scores.
- **AUC:** how well scores rank pneumonia above normal across possible thresholds. It is not the accuracy at our chosen threshold.

P7 reports **96.39% accuracy, 93.28% precision, 99.62% recall and 99.34% AUC**. Slide 6 compares the paper with our final result: accuracy {pct(shown['accuracy'])}, precision {pct(shown['precision'])}, recall {pct(shown['sensitivity'])} and AUC {pct(shown['auc'])}. Our additional measured metrics are specificity {pct(shown['specificity'])} and macro F1 {shown['macro_f1']:.3f}; slide 6 leaves them out because the paper does not report directly comparable values.

Say: **“Our ensemble correctly classified 92.15% of the 624 test images. Precision is close to the paper, but accuracy and pneumonia recall remain lower. This is our verified five-epoch baseline, not an exact match to the paper.”**

For questions, the saved confusion matrix (not displayed on slide 6) means {shown['confusion_matrix'][0][0]} normal images correctly recognized, {shown['confusion_matrix'][0][1]} false pneumonia predictions, {shown['confusion_matrix'][1][0]} missed pneumonia images and {shown['confusion_matrix'][1][1]} pneumonia images found. Rows are the true labels; columns are the predicted labels.

The audit checked every test image through every model again. It found no stale-feature, label-order or metric-arithmetic explanation for the second run's low score. Normal-case generalization was poor, and saved-view cycling showed systematic validation instability. The exact cause is not proved. Filename groups are patient proxies, not verified patient identities. The benchmark was inspected earlier, so we cannot call it a blind external holdout.

We cannot claim exact paper equivalence: the paper used 50 to 200 epochs and leaves several settings unpublished. Its reported headline metrics also do not fully reconcile to one confusion matrix on its stated test counts. That is a reproducibility question, not proof that the authors cheated. A high validation score does not establish a high test score.

## Our M3 proposal and next term

Our proposed final system uses **both lung-focused cropping and a CNN plus ViT ensemble**. Neither improvement has been implemented or measured yet.

First, use a fixed pretrained segmentation model to locate both lungs. Our candidate is TorchXRayVision's ChestX-Det PSPNet, which already has left-lung and right-lung output channels. Segmentation means assigning image pixels to an anatomical region. Crop the rectangle containing both lungs with a 10% margin, then resize it for classification. Crop original-resolution images before resizing: enlarging our existing small cache cannot recover lost detail. Failed crops fall back to the full image and are counted. The hypothesis is that more input pixels will represent lung tissue and fewer will represent peripheral areas. A bad crop can remove useful information.

Second, add an ImageNet-pretrained small Vision Transformer, with DeiT-Tiny as the resource-bounded candidate. A ViT splits an image into patches. Self-attention lets patches influence one another, potentially adding predictions that complement the CNNs. It does not automatically understand medicine or create new clinical evidence. This candidate is smaller than P9's DeiT-Base and is not an exact P9 reproduction.

We combine probability scores rather than six hard votes, which can tie. Choose the transformer weight and threshold using validation only. CNN-only controls use matching probability aggregation so a changed voting rule cannot masquerade as a ViT benefit.

The proposal has two components. Cropping is the primary change requested by the handout; ViT is an explicit second extension. We test both together without pretending they are one variable:

| Planned run | Input | Classifiers |
| --- | --- | --- |
| A | Full X-ray | Five CNNs |
| B | Lung crop | Five CNNs |
| C | Full X-ray | CNNs plus ViT |
| D | Lung crop | CNNs plus ViT |

B versus A tests cropping. C versus A tests ViT. D versus B tests ViT on crops. D versus C tests cropping with the hybrid. D is our intended final system. These letters label future experiments, not the three baseline runs already described.

Our combined prediction for D versus A is **macro F1 +0.01 to +0.04 and accuracy 0 to +3 percentage points**, with pneumonia recall no more than 2 points lower. These are the exact targets on the deck. They are predictions, not achieved gains, and the effects need not add together. No macro-F1 gain or a recall drop beyond 2 points would fail to support the combined proposal.

M4 will run these comparisons with a matching split and training settings, inspect failed crops and errors, and measure runtime. M5 will write up the results, including failures. JEPA feature transfer and mixup are optional later ideas, not this submission's implemented method. JEPA predicts representations of missing image regions; mixup blends training examples and labels. Neither guarantees better pneumonia recognition.

## What the demo shows

Choose a test sample or upload a JPEG/PNG and click Analyze image. The demo computes real CNN predictions and a ResNet18 Grad-CAM heatmap. Grad-CAM highlights regions contributing to one model's chosen class. Its map is coarse, not a proven pneumonia boundary or an explanation of the entire ensemble. Warmer colours indicate greater relative activation, not medical certainty.

The current demo uses the original verified CNN weights, as the deck says. It has no crop or ViT yet. The displayed mean pneumonia score is uncalibrated: 99% model output is not a verified 99% chance that a patient has pneumonia. Image-left and image-right heatmap shares do not establish the patient's anatomical side.

## A short explanation anybody can give

“We study normal-versus-pneumonia classification on pediatric chest X-rays. We selected Chouhan's five-CNN ensemble and ran three versions to inspect reproduction differences. The completed experiments showed strong pneumonia recall but too many false positives on normal images. The third version removes feature caching, restores the full paper-training partition and uses fresh augmentation. We report its score only after verification. For M3, we propose lung cropping plus a pretrained small transformer. Four controlled comparisons will test their separate and combined effects. The predicted gains are not results, and the demo is a research prototype.”

## The deck connection

The deck follows the same order: project and milestones; literature; P7 and dataset; three baseline runs; actual errors and paper comparison; crop plus ViT proposal; A/B/C/D comparisons and prediction; live demo and next steps. Everyone should know this whole story. We do not divide ownership by slide number.

If asked whether the model improved, use the verified score table above. If asked whether cropping or ViT helped, answer “We have not measured that yet.” If asked whether this matches all author settings, answer “We match recoverable details and disclose the remaining assumptions and shorter schedule.” Never fill a missing result with a guess.

Sources for checking, not extra required reading: [P7](https://doi.org/10.3390/app10020559), [P9](https://pmc.ncbi.nlm.nih.gov/articles/PMC10827725/), [existing hybrid](https://pmc.ncbi.nlm.nih.gov/articles/PMC10248870/). The accompanying deck is Reading_Room_M2_M3.pdf or its editable PPTX version.
'''
paths=[OUT/'TEAM_SHARED_PREP.md']+[OUT/f'{name}_M2_M3_PREP.md' for name in ['SAMARTH','MEGHAVI','RACHIT','HARSHAL']]
for p in paths:p.write_text(text)
digests={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};assert len(set(digests.values()))==1
(ROOT/'.build/shared_guide_receipt.json').write_text(json.dumps({'identical_content':True,'files':digests,'words':len(text.split()),'online_verified':latest is not None},indent=2))
print('Shared guide and four identical copies created.',len(text.split()),'words. Online verified:',latest is not None)
