<!-- FINAL_M4_STATUS -->
# Final milestones completed: declared frozen-feature pilot

The new lung-crop/ResNet18/DeiT-Tiny experiment and final presentation are in `../m4_m5/`. Start with `../FINAL_START_HERE.txt`. The demo now loads the final full FP32 M2 checkpoints and provides a separate live M4 crop-comparison mode. The new experiment is a reduced single-CNN frozen-feature pilot, with a documented 256-pixel segmentation departure. The M2/M3 material below records the earlier submission and its then-future proposal. The timestamped hypothesis remains unchanged.

---

# M2 and M3 submission

The required evaluation files are in `output/`:

- `M2_Reproduced_Baseline.ipynb`: executed notebook containing training code, actual results, confusion matrix, ROC curve, training histories, error examples and verification.
- `Reading_Room_M2_M3.pptx`: editable nine-slide update of the Reading Room presentation, with introduction, M2 results and M3 prediction.
- `Reading_Room_M2_M3.pdf`: the same nine-slide presentation for sharing or projecting.
- `M2_M3_Project_Report.docx`: introduction, literature summary, baseline result and proposed experiment.
- `TEAM_SHARED_PREP.md`: one common plain-language guide for all four members, aligned with the nine-slide deck.
- `M2_M3_Submission_Package.zip`: compact bundle including deliverables, source code, manifests and per-image results.

## Selected work

Base paper: Chouhan et al.2020, five pretrained CNNs with frozen EARLY layers, trainable LATE blocks/classifiers and majority voting. P7 was selected over the larger P9 transformer. The tentative M1 base changes while the lung-focused proposal continues. Our independent reconstruction has a declared five-epoch budget, not the full paper schedule. `REPRODUCIBILITY_CHECKLIST.md` lists assumptions and departures.

M3 final proposal: lung cropping plus a pretrained small ViT in the CNN ensemble. Cropping remains the primary handout hypothesis and ViT is an explicit second extension. Four matched A/B/C/D comparisons isolate full/cropped input and CNN-only/CNN-plus-ViT predictions with matching probability aggregation. **Neither improvement has been implemented or measured.** JEPA and mixup are optional future work.

The final corrected online five-epoch experiment is complete and independently verified. Submit `output/M2_Final_Online_Baseline.ipynb` as the primary results notebook. Final624-record test scores are92.15% accuracy,92.95% precision,94.62% recall,88.03% specificity, macro F1.916 and AUC.962. Its confusion matrix is[[206,28],[21,369]]. All5232 paper-training records trained for five fixed epochs, with fresh augmentation and full FP32 image forwards. Three models trained on the Mac and two on the isolated CPU VPS; all final evaluation and reload verification ran on the Mac. No test-driven selection.

Earlier experiments remain in the project: initial cached five epochs87.18% accuracy/83.12% precision, cached15-epoch audit85.74%/81.55%. The latter had98.94% validation accuracy that did not transfer to test. The third run improves accuracy/precision but has lower pneumonia recall and AUC than the cached audit. Multiple settings changed, so this is not an epoch-only ablation or proof of one cause. The updated deck/report/shared guide use final verified values.

## Read the results

`results/summary.json` contains measured ensemble metrics and image-level bootstrap intervals. `results/test_predictions.csv` records every original test record, label, five probabilities and vote count. Each individual model has training history, measured metrics and a saved trained head. `results/verification.json` independently recomputes the metrics and checks learned heads against their saved predictions on this Mac.

The ZIP intentionally excludes the large virtual environment, images, backbone weights, learned AlexNet head and feature caches. It retains probabilities, CSV results, figures and the executed notebook. All large measured artifacts remain in the local project folder. On another computer, the notebook can inspect saved scores without these caches; learned-head verification requires a deliberate reproduction run.

## Run locally on this Mac

The prepared environment is `.venv`. Current data and feature caches allow resumption without starting again.

```text
.venv/bin/python prepare_data.py
nice -n 15 .venv/bin/python baseline.py
.venv/bin/python verify_results.py
```

The corrected run uses CPU, two threads, frozen-prefix batches4, late-block training batches8, five fixed epochs, pauses, one model at a time and memory/disk guards. Figures4-8 specify trainable late blocks and a9216->512->2 AlexNet head; these are reconstructed. Earlier head-only probes are retained separately under results/preliminary_head_only and are not the final baseline. Five epochs is shorter than the paper's schedule. Do not launch multiple baseline processes concurrently.

## Local live demo

Run `.venv/bin/python -m uvicorn demo.app:app --host 127.0.0.1 --port 8765` from this folder, then open http://127.0.0.1:8765. Test images and upload both call the actual five learned models. A ResNet18 Grad-CAM overlay explains that component for the displayed class. It is not an ensemble-wide attribution or lesion localization. Input images are processed locally. The compact ZIP requires deliberate reproduction before live inference because it omits large learned model/cache files; the fully prepared Mac project already contains them.

## Reproduce from the compact bundle

Create a Python 3.11 environment and install `requirements.txt`. Open the notebook with that environment. Run it from the unpacked `M2_M3` folder (or its `output` folder). The default mode is `inspect_saved`. Change `MODE` to `reproduce` only when you intend to download approximately 1.23 GB of source shards, the official pretrained backbones, and retrain. The data preparation keeps one shard at a time and caches only the initial lossless 224-pixel resize. The notebook contains the preparation and model code; source `.py` files are also included.

Training may take substantial time on CPU. The resource guards stop if the process exceeds 2.6 GiB resident memory, system available memory falls below 650 MiB, or disk falls below 700 MiB. Cache progress survives interruption. Do not disable safeguards to force a run.

## Defend the result accurately

Say “our independently reconstructed baseline achieved the measured values in the notebook.” State the paper target and actual gap. Do not say the authors are wrong or that high AUC proves clinical validity. Small data and imbalance do not automatically mean low F1 or overfitting. P9 already reports F1 and confidence intervals, and CNN-transformer hybrids already exist.

The M2 paper-comparison protocol merges the original 16 validation records into training. M4 will create a proper group-aware validation split and rerun the full-image control before comparing the crop condition. A score from that new split cannot be compared directly to today's M2 score as an isolated crop effect.

There is no numerical marking rubric in the supplied handouts. Every explicit M2/M3 deliverable is covered, but marks depend on the evaluation and defence.
