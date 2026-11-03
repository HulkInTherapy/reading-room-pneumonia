# M3 hypothesis and prediction

Recorded 5 October 2026. Status: proposed experiment, not implemented or measured.

## Selected baseline

P7: Chouhan et al., A Novel Transfer Learning Based Approach for Pneumonia Detection in Chest X-ray Images, Applied Sciences 10(2), 559, 2020. DOI: https://doi.org/10.3390/app10020559.

Five ImageNet-pretrained CNNs (AlexNet, DenseNet121, InceptionV3, ResNet18 and GoogLeNet) use frozen early feature layers and trainable late blocks/classifiers as specified by the paper's Figures4-8. Three or more pneumonia votes give the final label. Our local five-epoch schedule is a declared resource limit. This replaces the tentative Kermany/InceptionV3 baseline presented in M1 while retaining the change to image region.

## One change

Replace the full image with a lung-focused crop derived from a pretrained chest X-ray lung segmentation model. Use the union of both lung masks, expand its bounding box by a fixed 10% margin, then resize the crop to 224 by 224. Keep the same five classifiers, training schedule, augmentation, split, seed and majority-vote rule in the paired control and crop experiments.

For M4, obtain original-resolution pixels from the verified source shards again and crop BEFORE the initial 224-pixel resize. The current Mac cache contains only the M2 initial resize. Cropping that small cache and enlarging it would not recover lung detail. Process original images one shard at a time rather than storing the entire full-resolution collection. The paired full-image control will use those same source pixels and the same final resize.

Cropping is one input-preprocessing variable. It does not introduce a transformer, a new loss or a different voting rule. The segmentation model must be fixed and trained on separate data. Failed masks fall back to the full image, with the failure count reported. A crop is a bounding rectangle and can contain non-lung pixels. We do not claim that it removes all shortcuts.

The concrete candidate is TorchXRayVision's pretrained ChestX-Det PSPNet, whose official code exposes Left Lung and Right Lung channels: https://github.com/mlmed/torchxrayvision. Its documentation, source and segmentation demo have been inspected; its weights and crop experiment have not been run here. It outputs 14 anatomy channels at 512 pixels and comes from ChestX-Det, not Kermany. M4 will use its documented preprocessing, apply sigmoid to the output logits as in the official demo, and threshold the left/right lung probabilities at 0.5. Pediatric mask quality remains an explicit limitation.

## Hypothesis and mechanism

Lung-focused cropping may improve the balance between normal-class and pneumonia-class recognition by devoting more of the fixed input resolution to lung tissue and reducing peripheral image variation. It may also remove useful anatomy or fail on pediatric images, so improvement is uncertain.

## Numerical prediction before the experiment

Relative to an identically run full-image control, we predict **macro F1 increases by 0.01 to 0.03**, **normal-class recall (specificity) increases by 2 to 5 percentage points**, and **ROC AUC changes by 0.00 to +0.02**. We predict pneumonia recall changes within **minus 2 to plus 2 percentage points**, and accuracy within **minus 1 to plus 2 percentage points**. These are reasoned predictions, not results, confidence intervals or a claim of state of the art.

This updates M1's prediction of less than one percentage point accuracy movement: after changing the base paper, we make macro F1 and specificity the primary expected benefit. We retain the possibility of little or no accuracy improvement. No crop experiment has been run for M2/M3.

## M4 evaluation design

1. Construct one fixed training/validation split from original non-test data, grouping recoverable patient identifiers and removing exact duplicate overlap. Document any normal-image identifier ambiguity. Keep the original 624-image test set sealed for paired final evaluation. If duplicated test records are retained for paper comparability, also report a unique-image sensitivity analysis. Do not tune on test scores.
2. Re-run the full-image control under that split. The M2 paper-comparison score is not interchangeable with this new protocol's control score.
3. Run the lung-crop version with exactly the same settings. Select checkpoints by validation macro F1; break ties by higher validation AUC, then earlier epoch. Keep the component threshold0.5 and three-of-five vote fixed. Use no test scores for selection, then evaluate the paired final conditions on test.
4. Repeat paired runs for seeds 42, 43 and 44 if resources allow. Report the seed results and paired bootstrap intervals for metric differences. Do not infer significance from overlap of separate intervals.
5. Report accuracy, pneumonia sensitivity, specificity, precision, both class F1 scores, macro F1, AUC and confusion matrices. Include five to ten errors and segmentation failures.
6. Optional diagnostic ablation: apply the fixed crop at test time only to the full-image baseline. This is a distribution-shift test, not the main treatment. A peripheral-only stress test cannot by itself prove shortcut learning.

Grad-CAM examples can support qualitative discussion for individual CNNs. Heatmap overlap with a lung mask measures anatomical concentration, not correct lesion localization, clinical correctness or causal reasoning. There are no lesion annotations here.

## What would refute the prediction

No macro-F1 gain, substantially worse pneumonia recall, or a crop effect indistinguishable from zero across paired seeds would fail to support the hypothesis. We will report the measured outcome even if cropping hurts. The assignment explicitly allows a wrong prediction when reasoning and analysis are sound.

## Scope

This is a student research-reproduction project on a single-center pediatric benchmark. We make no clinical deployment claim and no novelty or state-of-the-art claim. M3 ends with this proposal and the report introduction. Implementation and ablations belong to M4 after submitting this hypothesis.
