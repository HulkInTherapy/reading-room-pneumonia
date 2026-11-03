# M3 lung cropping and CNN ViT proposal

## Revision before implementation

Revised5 October2026 before implementing either improvement. The proposed final system combines lung cropping and a pretrained small ViT with the five CNNs. Neither improvement has been implemented or measured. All three baseline experiments use full images and CNNs only. The corrected five-epoch experiment removes feature caching and restores the full paper-training partition. This revision supersedes the crop-only scope below, while retaining its crop mechanism, safeguards and numerical predictions.

The handout asks for one primary change. Cropping remains that primary hypothesis and the transformer is an explicit second extension. We will not describe their joint implementation as a one-variable experiment.

The final candidate is: a fixed lung segmenter supplies both lung masks; crop their union bounding box with10% margin before resizing original-resolution pixels to224; run the five CNNs and an ImageNet-pretrained DeiT-Tiny candidate; combine pneumonia probabilities. DeiT-Tiny is a resource-bounded choice, not an exact P9 DeiT-Base reproduction.

| Experiment | Input | Classifiers |
| --- | --- | --- |
| A | Full X-ray | Five CNNs, probability aggregation |
| B | Lung crop | Five CNNs, probability aggregation |
| C | Full X-ray | Five CNNs plus pretrained ViT |
| D | Lung crop | Five CNNs plus pretrained ViT |

B-A isolates cropping. C-A isolates adding ViT. D-B tests ViT on crops. D-C tests cropping in the hybrid. D is our final target, not the only comparison.

Use score=(1-w)*mean(CNN pneumonia probabilities)+w*ViT pneumonia probability. Predeclare w in{0,0.1,0.2,0.3,0.5} and thresholds in{0.4,0.5,0.6}. Select using validation macro F1 subject to the recall safeguard, then AUC, then lower ViT weight. For equal weight/AUC, prefer threshold0.5, then the lower threshold. The CNN-only comparator uses the same probability aggregation and validation threshold-selection process with w=0. Report the original P7 majority-vote reconstruction separately. Changing aggregation must not be mistaken for a ViT benefit.

Retain the crop prediction below for B-A. For adding ViT under a matched aggregation protocol, predict macro F1 +0.01 to +0.03 and accuracy +1 to +3 percentage points, with zero or negative gain possible. For D-A, predict macro F1 +0.01 to +0.04 and accuracy0 to +3 points with pneumonia recall no more than2 points lower. These are hypotheses, not measured gains, and are bounded by metric maxima. Crop and ViT gains must not be added together. No macro-F1 gain or a recall loss beyond2 points fails to support the combined hypothesis.

Keep the same deduplicated group split, source images, comparable per-model schedules, CNN trainable boundaries and augmentation across matched conditions. Choose checkpoints, weights and thresholds on validation only, never test. Filename groups remain proxies and the original benchmark test was previously inspected. Report class-specific metrics, paired uncertainty, crop failures and runtime. Begin ViT with a frozen-backbone classifier pilot and consider limited last-block tuning only after memory/time checks. One model at a time, two CPU threads and existing resource guards remain mandatory. No crop/ViT training starts before this M3 proposal is submitted.

CNN-transformer hybrids already exist. JEPA feature transfer and mixup remain optional later experiments, outside the primary four-way comparison. I-JEPA predicts image representations; V-JEPA focuses on video/world modelling. Mixup blends examples and labels and is not LeCun's JEPA method. None supplies new independent clinical evidence.

Sources: [DeiT](https://github.com/facebookresearch/deit), [existing hybrid](https://pmc.ncbi.nlm.nih.gov/articles/PMC10248870/), [I-JEPA](https://github.com/facebookresearch/ijepa), [mixup](https://arxiv.org/abs/1710.09412).

Recorded 5 October 2026. Status: proposed experiment, not implemented or measured.

## Selected baseline

P7: Chouhan et al., A Novel Transfer Learning Based Approach for Pneumonia Detection in Chest X-ray Images, Applied Sciences 10(2), 559, 2020. DOI: https://doi.org/10.3390/app10020559.

Five ImageNet-pretrained CNNs (AlexNet, DenseNet121, InceptionV3, ResNet18 and GoogLeNet) use frozen early feature layers and trainable late blocks/classifiers as specified by the paper's Figures4-8. Three or more pneumonia votes give the final label. Our local five-epoch schedule is a declared resource limit. This replaces the tentative Kermany/InceptionV3 baseline presented in M1 while retaining the change to image region.

## Primary crop hypothesis retained from the earlier proposal

Replace the full image with a lung-focused crop derived from a pretrained chest X-ray lung segmentation model. Use the union of both lung masks, expand its bounding box by a fixed 10% margin, then resize the crop to 224 by 224. Keep the same five classifiers, training schedule, augmentation, split, seed and majority-vote rule in the paired control and crop experiments.

For M4, obtain original-resolution pixels from the verified source shards again and crop BEFORE the initial 224-pixel resize. The current Mac cache contains only the M2 initial resize. Cropping that small cache and enlarging it would not recover lung detail. Process original images one shard at a time rather than storing the entire full-resolution collection. The paired full-image control will use those same source pixels and the same final resize.

Cropping is one input-preprocessing variable. In B versus A, keep the classifier family and aggregation identical. C and D separately introduce the transformer extension stated above. The segmentation model must be fixed and trained on separate data. Failed masks fall back to the full image, with the failure count reported. A crop is a bounding rectangle and can contain non-lung pixels. We do not claim that it removes all shortcuts.

The concrete candidate is TorchXRayVision's pretrained ChestX-Det PSPNet, whose official code exposes Left Lung and Right Lung channels: https://github.com/mlmed/torchxrayvision. Its documentation, source and segmentation demo have been inspected; its weights and crop experiment have not been run here. It outputs 14 anatomy channels at 512 pixels and comes from ChestX-Det, not Kermany. M4 will use its documented preprocessing, apply sigmoid to the output logits as in the official demo, and threshold the left/right lung probabilities at 0.5. Pediatric mask quality remains an explicit limitation.

## Hypothesis and mechanism

Lung-focused cropping may improve the balance between normal-class and pneumonia-class recognition by devoting more of the fixed input resolution to lung tissue and reducing peripheral image variation. It may also remove useful anatomy or fail on pediatric images, so improvement is uncertain.

## Numerical prediction before the experiment

Relative to an identically run full-image control, we predict **macro F1 increases by 0.01 to 0.03**, **normal-class recall (specificity) increases by 2 to 5 percentage points**, and **ROC AUC changes by 0.00 to +0.02**. We predict pneumonia recall changes within **minus 2 to plus 2 percentage points**, and accuracy within **minus 1 to plus 2 percentage points**. These are reasoned predictions, not results, confidence intervals or a claim of state of the art.

This updates M1's prediction of less than one percentage point accuracy movement: after changing the base paper, we make macro F1 and specificity the primary expected benefit. We retain the possibility of little or no accuracy improvement. No crop experiment has been run for M2/M3.

## M4 evaluation design

1. Use the fixed deduplicated group split from the audited restart or predeclare any justified replacement before the new runs. Document normal-image identifier ambiguity. Preserve the original624-image benchmark partition for paired final evaluation, but disclose it was inspected earlier and is not a blind external holdout. Also report unique-image sensitivity analysis. Do not tune on test scores.
2. Re-run the full-image control under that split. The M2 paper-comparison score is not interchangeable with this new protocol's control score.
3. Run A/B/C/D with matching settings and the probability-aggregation controls specified in the revision. Select checkpoints by validation macro F1, then AUC and earlier epoch. The original three-of-five vote remains a separately reported reconstruction, not the aggregation control for judging ViT. Select any proposed weights/thresholds using validation only, then evaluate all declared conditions on test.
4. Repeat paired runs for seeds 42, 43 and 44 if resources allow. Report the seed results and paired bootstrap intervals for metric differences. Do not infer significance from overlap of separate intervals.
5. Report accuracy, pneumonia sensitivity, specificity, precision, both class F1 scores, macro F1, AUC and confusion matrices. Include five to ten errors and segmentation failures.
6. Optional diagnostic ablation: apply the fixed crop at test time only to the full-image baseline. This is a distribution-shift test, not the main treatment. A peripheral-only stress test cannot by itself prove shortcut learning.

Grad-CAM examples can support qualitative discussion for individual CNNs. Heatmap overlap with a lung mask measures anatomical concentration, not correct lesion localization, clinical correctness or causal reasoning. There are no lesion annotations here.

## What would refute the prediction

No macro-F1 gain, substantially worse pneumonia recall, or a crop effect indistinguishable from zero across paired seeds would fail to support the hypothesis. We will report the measured outcome even if cropping hurts. The assignment explicitly allows a wrong prediction when reasoning and analysis are sound.

## Scope

This is a student research-reproduction project on a single-center pediatric benchmark. We make no clinical deployment claim and no novelty or state-of-the-art claim. M3 ends with this proposal and the report introduction. Implementation and ablations belong to M4 after submitting this hypothesis.
