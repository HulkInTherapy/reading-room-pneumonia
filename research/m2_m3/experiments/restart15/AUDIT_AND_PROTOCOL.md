# Reproduction audit and fresh 15 epoch experiment

The existing five-epoch results and demo checkpoints are preserved. This experiment is a fresh start from ImageNet, not a continuation of those weights.

## Confirmed discrepancies and limits

1. Table 2 lists ResNet18 and AlexNet at 200 epochs, DenseNet121/InceptionV3 at 100, and GoogLeNet at 50. The prose additionally says AlexNet used 200 **iterations** at LR0.001, followed by LR0.00001, without explaining how that phase relates to the table's 200 epochs. Local runs used five epochs. The earlier audit incorrectly called ResNet's duration unspecified; the table resolves it to a reported 200 epochs. Fifteen epochs remains a deliberately truncated reproduction, not the complete published schedule.
2. The previous run used only two fixed cached image views per example. The paper describes augmentation and added noise, but does not publish a reproducible transform/noise configuration. Repeated fixed views limit variation and can encourage memorization.
3. The previous training used batches of eight and updated late-block BatchNorm running statistics. The paper does not disclose its batch size or BatchNorm policy. This is an unverified implementation assumption with potential train/eval consequences, not a proven explanation for the gap.
4. Frozen activations used float16. Its numerical effect must be checked rather than assumed to cause a ten-point difference.
5. There was no independent validation set after combining train/val to match the paper. The last epoch was evaluated without a validation-based checkpoint choice.
6. The 624-record test set has a different class prevalence from the training set. Prior shift and uncalibrated scores can affect a fixed decision threshold. Test prevalence will not be used to tune thresholds or loss weights.
7. DenseNet norm5 stays frozen; AlexNet activation/dropout choices are assumptions. The paper figures specify the late trainable boundaries, but absent author code prevents verifying these details.
8. Paper ROC-score construction, seed, batch size, exact normalization, augmentation severity and checkpoint-selection procedure are missing. Its prose has inconsistent counts, although Table1's 5232/624 counts match the downloaded mirror. We cannot claim these unspecified settings have been exactly reproduced.
9. Our cached frozen prefixes run in evaluation mode, which also freezes early BatchNorm running statistics. Freezing model weights does not automatically freeze BatchNorm statistics in an ordinary end-to-end training loop. The paper does not disclose that policy. Prefix caching therefore adds a potentially relevant departure if the authors allowed those statistics to update. The old late-tail BN diagnostic does not test this early-prefix issue; its near-perfect scores use images the old model already trained on and are not independent validation evidence.
10. The paper specifies an initial224x224 RGB resize, which matches our cache. Exact interpolation, ImageNet weight revision and transform defaults remain unverified. Our crop scale0.8–1.0 must not be mistaken for recovered author parameters. Matching the dataset's counts and common partition does not independently verify the authors' exact file list or image bytes because no executable split manifest was supplied.

The strong AUC with 79 false positives and one false negative suggests that ranking quality and the operating decision point need separate analysis. It does not establish the cause or prove the paper invalid.

## Additional cross split checks

The original-source hashes have no non-test/test overlap. Training and validation have neither shared source hashes nor shared recoverable filename groups. An additional 63-bit DCT perceptual screen (Hamming distance at most4, normalized32x32 grayscale mean absolute difference at most0.03) found zero close-image candidates across train/validation and non-test/test. The screen is approximate and cannot certify patient independence. Filename groups, especially normal-image identifiers, are proxies rather than verified patient IDs.

Simple cached-image brightness/border summaries do not expose one dramatic normal-image shift. Resolution differs by class in the source data, and the cache reflects the paper's initial224x224 resize. These descriptive summaries do not rule out hospital, acquisition, annotation or disease-severity differences.

## Conditional inconsistency in the published target

If Table 2's precision93.28%, recall99.62% and accuracy96.39% come from one binary confusion matrix on Table1's390 pneumonia/234 normal test images, they do not agree. Precision/recall imply approximately388.52 true positives,27.99 false positives and accuracy95.28%, not96.39%. Integer counts on624 images also cannot round to96.39% accuracy. Separate-run averages, alternative metric definitions, a different evaluation partition or reporting errors could explain this. The paper does not provide enough executable evidence to resolve it. This is not proof of fabrication and does not by itself explain our remaining gap. The exact96.39% target should not be treated as independently verified.

## Numerical check of our caches

For16 balanced non-test examples per model, float16 activation conversion changed probabilities by at most0.000055 and changed no0.5-threshold decisions. That sampled result argues against quantization as the dominant ten-point-gap cause; it is not a guarantee for every image.

## Precommitted restart protocol

- Deduplicate non-test images by original source SHA256; conservatively group recoverable patient filename prefixes. Use one fixed stratified group fold for approximately20% validation. No identical source hash or group may cross train/validation. Normal-image identifiers remain filename proxies rather than verified patient IDs.
- Preserve original624-record test partition. It has already been inspected in earlier work and is therefore not a pristine blind holdout. No test predictions, labels or metrics will select settings/checkpoints/thresholds in this restart.
- Reset every model to its official ImageNet weights and reset newly initialized classifiers and Adam state. Keep the paper's late-block training boundaries.
- Train15 epochs with batch16 and Adam. For the four non-AlexNet models, use LR0.001 halved every three epochs. For AlexNet, follow the stated200 optimizer iterations at0.001, then0.00001 for the remaining bounded15-epoch run. The paper's missing second-phase duration remains unknown. Batch16 is a local resource choice, not a recovered paper setting.
- Use four deterministic cached views: one clean, three independently seeded augmented views. The augmented transforms use horizontal flip, resized crop scale0.8–1.0, brightness/contrast0.15, plus Gaussian pixel noise std0.01 before normalization. Noise magnitude is an explicit local assumption. This improves variation but is still not fresh per-epoch augmentation.
- A disk resource guard stopped the first attempted epoch before completion. To reduce page-cache/swap pressure, the restart streams one view per epoch, cycling clean/aug1/aug2/aug3, and releases inactive mapped pages when supported. No guard was disabled; no completed epoch or test result was discarded. This is another declared caching approximation.
- Keep float16 frozen-prefix caches only if a controlled numerical check finds negligible effect; convert to float32 for trainable layers. Preserve ordinary BatchNorm training in the late blocks for the primary run; increasing batch size addresses small-batch stability without inventing an author freeze policy. Keep DenseNet norm5 frozen as before and disclose it.
- Report validation loss, both class recall/F1, macro F1 and AUC each epoch. Save epoch5/10/15 and optimizer/scheduler/RNG state. Select each model by validation macro F1 at fixed0.5 threshold, tie-break higher AUC, then earlier epoch.
- Primary ensemble remains three-of-five votes with fixed component threshold0.5. An optional threshold variant may use validation only and must be labelled a separate calibration experiment, not the paper baseline.
- Evaluate the selected ensemble once on the original test records, and also report the618-unique-image sensitivity analysis. Compare the new matched-protocol epoch5/10/15 validation results to judge the effect of more training. Comparing to the old run does not isolate epochs because the split/batch/views also changed.
- One model at a time, two CPU threads, process RSS guard2.6GiB, available-memory guard650MiB and disk guard700MiB. Preserve the current demo until the new run is verified.

Lung cropping remains the separate M3 proposal and is not implemented in this audit or restart.
