# Live demo walkthrough

1. Open http://127.0.0.1:8765, or double-click START_DEMO.command in the prepared Mac project. Keep the local server terminal open. The trained models and input images are already on this machine, so the final demo needs no cloud account.
2. Select a test case. Click Analyze image. Wait for the real five-model inference and ResNet18 Grad-CAM computation to finish.
3. Say: “This label comes from at least three of the five CNN votes. The displayed mean pneumonia score is a separate, uncalibrated score. The known test label appears only after inference so we can check the prediction.”
4. Point to the heatmap. Say: “This is Grad-CAM from the ResNet18 component for the displayed class. It uses actual gradients and activations. It is a coarse seven-by-seven map enlarged to the input size. It does not show proven lesion boundaries or the reasoning of the entire ensemble.”
5. Upload a JPEG or PNG X-ray. Click Analyze image again. Uploads do not have a known ground-truth label.
6. If a prediction is wrong, show it honestly. The notebook has the full confusion matrix and error examples. A high model score can still be wrong.

## What we built and what remains proposed

Built: a dataset audit, five-CNN reconstruction, original five-epoch run and separate audited15-epoch restart, saved per-image outputs and independently verified notebooks. The live demo retains the original five-epoch CNN weights. Its scores therefore belong to that version, not the restart.

Proposed: lung-focused cropping plus a pretrained small ViT in the CNN ensemble, with four matched controls. Neither improvement has run. M3 records predictions before implementation. The demo still uses full X-rays and five CNNs only. Grad-CAM is a visualization, not segmentation. JEPA and mixup remain optional later work.

## Limits to state clearly

The training budget is shorter than the paper's schedule. The data are pediatric and single-center. The image-left and image-right activation shares do not establish a patient's anatomical side. This is a research demonstration, not clinical diagnosis. We do not claim a new state of the art.

## Recovery during the presentation

If the browser disconnects, reopen the local URL. If the server is stopped, double-click START_DEMO.command. Do not start another training job. A saved screenshot of a real tested result is included in the deck, but it is labeled as a screenshot rather than live inference.
