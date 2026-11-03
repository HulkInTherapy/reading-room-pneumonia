# Data and pretrained components

This repository combines original student project code and artifacts with the following public resources. Their original terms continue to apply.

- **Pediatric chest X-ray data:** Kermany et al., Mendeley Data v2, DOI [10.17632/rscbjbr9sj.2](https://doi.org/10.17632/rscbjbr9sj.2), accessed through the [hf-vision mirror](https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia). Samples shown in the demo and figures come from this public dataset. The full training collection is not included in Git.
- **ResNet18 and historical CNN weights:** official PyTorch/TorchVision model releases. See [TorchVision](https://github.com/pytorch/vision).
- **DeiT-Tiny:** the ImageNet-pretrained DeiT model distributed through timm. See [DeiT](https://github.com/facebookresearch/deit) and [timm](https://github.com/huggingface/pytorch-image-models).
- **ChestX-Det PSPNet:** fixed pretrained anatomical segmentation through [TorchXRayVision](https://github.com/mlmed/torchxrayvision). It is used at the explicitly documented 256-pixel inference resolution.
- **Fonts:** Manrope and DM Mono. Their supplied OFL notices are retained beside the local font files in `web/assets/`.
- **Reference method:** Chouhan et al., _A Novel Transfer Learning Based Approach for Pneumonia Detection in Chest X-ray Images_, Applied Sciences 10(2), 559 (2020), DOI [10.3390/app10020559](https://doi.org/10.3390/app10020559).

The final report distinguishes our measured experiments from authors' published results and earlier superseded project claims. This repository is a research and educational demonstration, with no claimed clinical validation.
