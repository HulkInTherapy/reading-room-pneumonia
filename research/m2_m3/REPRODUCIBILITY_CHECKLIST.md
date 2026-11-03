# Reproducibility checklist

## P7 selected base paper

Chouhan et al., Applied Sciences 2020, 10(2), 559. https://doi.org/10.3390/app10020559

Full text inspected through the author repository PDF: https://eprints.qut.edu.au/199742/1/applsci_10_00559_1_.pdf.

| Item | Evidence and decision |
|---|---|
| Public code preferred | No paper-linked author repository verified in the paper or search. Our baseline.py is an independent reconstruction. PyTorch's official torchvision implementations supply the five backbones. |
| Public data | Kermany CXR, DOI 10.17632/rscbjbr9sj.2. Downloadable CC BY 4.0 mirror hf-vision/chest-xray-pneumonia. Original split counts 5216/16/624. Parquet source checksums checked before extraction. |
| Paper split | Table 1: 5232 training, 624 test, formed by combining original train and validation. Local baseline uses this partition and a fixed training budget. |
| Model identities | AlexNet, DenseNet121, InceptionV3, ResNet18, GoogLeNet, ImageNet-pretrained. Figures 4-8 specify frozen early layers and trainable late layers. Train ResNet layer4, DenseNet block4 layers15/16, Inception Mixed_7c and GoogLeNet inception5b plus classifiers. AlexNet replaces the classifier with 9216->512->2. Combine three-of-five votes. |
| Input | Paper starts with 224x224x3 resizing, then horizontal flips, random resized crops and varying intensity. Local initial resize is cached losslessly as PNG. |
| Objective and optimizer | Paper states Adam, cross entropy, initial LR 0.001, halved every three epochs. These are reconstructed locally. |
| Training duration departure | Paper Table2 lists ResNet18/AlexNet200 epochs, DenseNet/Inception100 and GoogLeNet50. AlexNet prose also describes200 iterations at0.001 then0.00001, ambiguously related to its epoch count. Local original baseline uses FIVE epochs for every model, with the same general learning-rate decay, and does not follow AlexNet's separate low-rate phase. The audited restart uses15 epochs with AlexNet's stated stepwise rate. Neither is full-schedule replication. |
| Augmentation departure | Local has one clean and one fixed seeded augmented view per training image, sampled during head training, rather than a fresh image augmentation every epoch. The paper also mentions added noise without giving a distribution/scale; local reconstruction does not add a separate noise transform. Frozen features permit caching, but these are approximations and can change results. |
| Other unspecified settings | Paper does not provide seed, exact environment, crop/intensity settings, normalization, batch size or optimizer details. Local: seed 42, frozen-prefix feature batch4, training batch8, ImageNet mean/std, crop scale0.8-1.0, brightness/contrast0.15, Adam defaults. Cached frozen activations use float16 for disk space and convert to float32 during training. |
| BatchNorm and dropout | Frozen prefixes use evaluation mode. Trainable late blocks use training mode. DenseNet's post-block norm5 remains frozen/eval. AlexNet's replacement 512-unit classifier uses ReLU/dropout0.5, whose exact hidden activation/dropout is not specified in its diagram. GoogLeNet/Inception retain their standard dropout. |
| Metrics | Paper reports ensemble accuracy 96.39%, recall 99.62%, precision 93.28%, AUC 99.34%. Local reports these plus specificity, both class F1, macro F1 and balanced accuracy. AUC uses mean CNN probabilities; class labels use majority vote. Paper's AUC score construction is not specified. |
| Validation/test handling | Original train+val merged to match Table1. Five corrected epochs are fixed before testing; no test-based checkpoint/threshold selection. An earlier fully frozen diagnostic probe was tested, then discarded as the main baseline after inspecting the figures. Its outputs are retained separately, not mixed into the final ensemble. No held-out validation subset exists in the M2 paper-comparison protocol. |
| Hardware | 8 GiB Apple M2. One model at a time, bounded feature extraction, two CPU threads, memory/disk guards. Paper used GTX1070 and 48 GB host RAM. |
| Claim boundary | This is a method reconstruction, not a bit-for-bit reproduction of unavailable code. A near metric does not erase departures. Report the actual reproduction gap. |

## P9 assessment

Singh et al. 2024: https://pmc.ncbi.nlm.nih.gov/articles/PMC10827725/.

The verified pretrained model is DEIT_Base_Patch16_224, not a generic student ViT repository. Table 5 reports 30 epochs, 97.61% accuracy, sensitivity .949, specificity .981, F score .952, and AUC .966. Table 6 uses slightly different F-score rounding. It also includes confidence intervals.

The article states 80/10/10 splitting (4684/586/586), while class totals in its split table contradict its dataset distribution table. No fixed seed, split manifest or author training repository was verified. Therefore exact reproduction is not established. Full 30-epoch fine-tuning of the 85.8M-parameter model is not the selected local deadline task.

## Novelty check

CNN/ViT hybrid and ensemble approaches already exist. For example: https://pmc.ncbi.nlm.nih.gov/articles/PMC10248870/ (2023). The project contribution is a controlled, predicted and tested preprocessing change to our own reproduced baseline, as the assignment requests. We do not claim that lung cropping or hybrid models are globally new.
