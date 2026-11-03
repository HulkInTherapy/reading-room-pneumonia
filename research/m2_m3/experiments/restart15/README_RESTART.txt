AUDITED FIFTEEN EPOCH RESTART

This folder is separate from the original five-epoch results. It contains the
audit, fixed split, executable training code, per-epoch histories and checkpoints.
The current demo still uses the original verified models.

LIVE STATUS
Open STATUS.json. "training" means results are not final.
"complete_and_verified" means all five 15-epoch runs finished, saved weights were
independently reloaded and the new inspection notebook executed successfully.
Other states name a failure; inspect finalization.log or the training output.

COMPLETION OUTPUTS
results/summary.json                  Measured test and validation metrics
results/independent_verification.json  Reload verification and checkpoint hashes
results/AUDIT_RESULTS.html            Human-readable comparison and real figures
../../output/M2_Audited_15_Epoch_Restart.ipynb  Executed results notebook

These files only exist after the corresponding work succeeds. No predicted score
is substituted for an actual result. The new notebook retains the distinction
between CLI training and notebook execution of verification/analysis cells.

DELIBERATE RESUME IF A RESOURCE GUARD STOPS TRAINING
From /Users/zebra/cvproj/m2_m3:
nice -n 15 .venv/bin/python -u experiments/restart15/run.py

Do not start a second worker while one is already running. Resume state contains
model, optimizer and RNG. Completed models are skipped. Resource guards are not
disabled. If disk or memory is still below the bound, resolve that condition first.
After a resumed run completes, run:
.venv/bin/python experiments/restart15/finalize.py

INTERPRETATION
The paper reports 50-200 epochs depending on model, not just 15. The old and new
experiments differ in training split, augmentation, batch size, AlexNet schedule
and validation checkpoint selection, so their comparison cannot isolate epochs.
Test was seen in earlier work. No test-score selection occurs in this restart.
M3 lung cropping is still a hypothesis, not an implemented or measured change.
