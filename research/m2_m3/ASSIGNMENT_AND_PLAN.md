# Milestones 2 and 3 submission plan

## Exact requirements from the two supplied documents

The handout distinguishes stages from milestones. Stage 2 selects a base paper and fills the reproducibility checklist. Stage 3 runs the baseline on the same data, with the same split logic, preprocessing and backbone where recoverable, and records the reproduction gap. **M2 is the reproduced baseline that runs.** Stage 4 / **M3** submits one change, its direction, rough numerical magnitude and mechanism before implementation, plus an introduction.

The teacher's evaluation message requires (1) a notebook containing code and reproduced results, (2) a hypothesis and brief introduction added to the earlier presentation, and (3) an introduction added to the literature-survey report. M4 covers the proposed change and ablations. M5 covers failure analysis, IEEE-format paper and defence. No clinical claims are allowed. Sensitivity and AUC must accompany accuracy, and a limitations section is required.

## Decision

Use **P7, Chouhan2020**, as the base paper, reconstructing its five pretrained CNNs and majority voting. Preserve the original five-epoch reconstruction and separate audited15-epoch restart. The proposed final path combines **lung-focused cropping and a pretrained small ViT**. Cropping remains the primary one-change hypothesis required by the handout, while the transformer is an explicit second extension. Four matched comparisons test full/cropped input with CNN-only/CNN-plus-ViT probability aggregation. Do not describe this two-component proposal as one variable. P9 provides transformer context, not an exact reproduced model; hybrids already exist. JEPA and mixup remain optional later work.

The Mac is Apple M2 with 8 GiB RAM, initially under 6 GiB free disk. All measured local feature extraction and training use CPU, two threads, small batches, pauses, one model at a time, and memory/disk guards. Cached features are resumable. No unrestricted long full-model run is authorized by this plan.

## M1 continuity and corrections

The M1 final slide tentatively chose Kermany 2018 / InceptionV3 and proposed lung segmentation/cropping, paired controls, sensitivity and AUC. The current request explicitly authorizes changing that tentative baseline to P7 or P9. Keep the preprocessing proposal and document why P7 is reproducible within available resources.

The full P9 article contradicts parts of the old slide deck: it reports F score 0.952 (Table 5), AUC 0.966, and a confidence-interval section. It uses DeiT-Base patch16 at 224 pixels with 30 epochs, Adam, cross entropy and learning rate 1e-5. It reports an 80/10/10 re-split, not the original 624-image test protocol. Its Table 3 class counts sum to 3,895 pneumonia / 1,961 normal, inconsistent with Table 2's 4,273 / 1,583. An author repository or fixed split manifest has not been verified. A student ViT GitHub repository is not author code for P9.

Do not reuse the old claims that all papers have no F1/CI or all evaluate exactly the same test images. Overlap of separate confidence intervals does not establish statistical equivalence or significance. Single-center origin and imbalance are limitations, not proof of bias, overfitting, low F1 or invalidity. An always-pneumonia classifier has 62.5% test accuracy and pneumonia F1 about 0.769, but normal-class recall zero. Macro F1 exposes that weakness.

## Deliverables

- Executed notebook with actual dataset audit, five model outputs, ensemble metrics, confusion matrix, ROC curve and declared reconstruction assumptions.
- Source code, dependency versions, source manifests, saved heads and per-image probabilities so results can be checked.
- Reproducibility checklist and paper-versus-local comparison, including unknown or changed settings.
- Timestamped M3 hypothesis with numerical predictions, not improved results.
- Report introduction and a report containing the existing literature topics.
- Updated earlier presentation with M2/M3 pages and a short evaluation deck.
- One short speaking guide for each team member.
- A local real-inference demo with sample selection, image upload and ResNet18 component Grad-CAM, plus browser/API verification and a double-click launcher.

## Marks alignment

No rubric with numerical marks appears in the supplied documents, so full marks cannot be guaranteed. This plan covers every explicit submission requirement. If a resource guard stops before all five models finish, report exactly which component ran and do not claim the ensemble or baseline target was reproduced.
