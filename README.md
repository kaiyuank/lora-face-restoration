# Efficient Facial Image Restoration Using LoRA-Based Fine-Tuning

Sam Kang | Washington University in St. Louis  
Advisor: Professor Ulugbek Kamilov

[Research report](report.pdf) | [Code guide](Code/README.md) | [Presentation slides](Presentation_Slides/)

Implemented LoRA personalization of the Gen2Res diffusion U-Net and compared dataset sizes, scaling, and attention, residual, and timestep-embedding adaptation using PSNR, SSIM, and LPIPS.

In the reported configuration, attention-only LoRA reduced fine-tuning time by 46% and GPU memory usage by 47%, with comparable perceptual quality. Per-identity storage decreased from 527 MB full checkpoints to 200 KB adapters, used with the shared base model.

This repository archives research code and presentation materials. Data and model checkpoints are maintained separately; execution status is described in the code guide.

Based on [Gen2Res](https://github.com/adobe-research/gen2res). Inherited code and its modifications are subject to the [Adobe Research License](Code/LICENSE).
