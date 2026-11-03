# Research evidence

Accessed 5 October 2026. Tavily searches were actually executed and saved as JSON under sources. The requested deep-research skill was found at /Users/zebra/.claude/skills/deep-research/SKILL.md after the supplied .Codex path did not exist. The specialized WebFetch tool is not exposed in this session, so full primary sources were fetched directly over HTTPS with curl after Tavily discovery. No search snippets alone were used to reconstruct the selected method.

## Primary sources inspected

- [Assignment handout](/Users/zebra/Downloads/CV_Student_Handout_Project%20%281%29.docx): complete document text extracted from OOXML. M2 baseline runs, M3 prediction before coding plus introduction, sensitivity/AUC and limitations.
- [Worked example](/Users/zebra/Downloads/CV_Sample_Flow.docx): complete document text extracted. Same data/protocol for replication, reproduction-gap analysis, one-variable hypothesis and numerical direction/magnitude.
- [P7 author-repository PDF](https://eprints.qut.edu.au/199742/1/applsci_10_00559_1_.pdf): full text and architecture Figures4-8 inspected. Frozen EARLY layers with TRAINABLE late blocks,9216->512->2 AlexNet classifier, five named CNNs, majority vote, augmentation, schedules,5232/624 split and Table2 metrics. Table2 reports200 epochs for ResNet18/AlexNet,100 for DenseNet/Inception and50 for GoogLeNet; AlexNet prose also describes200 iterations before lowering LR. Earlier fully frozen probes were corrected after reading these figures and are archived separately. Confidence HIGH for stated architecture boundaries. No author repository was verified; this does not prove none exists. The audited restart documents conditional metric inconsistencies and unresolved settings separately.
- [P9 full article](https://pmc.ncbi.nlm.nih.gov/articles/PMC10827725/): full text read, including Tables 2-6. DeiT-Base weights, F score, AUC, CI, 30 epochs and inconsistent split-class totals. Confidence HIGH for the presence of those statements; exact author split is unknown.
- [Dataset card](https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia): full card and file API inspected. Actual downloaded files were checked against LFS SHA256 and audited locally. Confidence HIGH for local record counts and hash checks. Mirror integrity does not certify clinical labels or patient independence.
- [Official torchvision repository](https://github.com/pytorch/vision): inspected as backbone implementation source. Installed torchvision 0.29.1 implementations and published default ImageNet checkpoints are used by our reconstruction, not represented as Chouhan's code.
- [Official DeiT README](https://github.com/facebookresearch/deit): full README fetched as architectural source, not represented as Singh's training code.
- [Hybrid explainable ensemble transformer encoder paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10248870/): full article fetched. Demonstrates that hybrid/ensemble transformer approaches to pneumonia existed before this proposed project. We make no global novelty claim.

## Important corrections to retained M1 material

P9 reports F score and confidence intervals. P9 does not use the original 624-image test set. The original claim that every later accuracy gain is statistical noise was not supported by paired comparisons and should not be defended. We use actual local estimates and explicitly acknowledge protocol differences. The revised evaluation deck removes those claims.
