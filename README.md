# Efficient Facial Image Restoration Using LoRA-Based Fine-Tuning

Sam Kang | Washington University in St. Louis  
Advisor: Professor Ulugbek Kamilov  
Presented as a poster at WashU ESE Day.

[Research report](report.pdf) | [Code guide](Code/README.md) | [Presentation slides](https://github.com/kaiyuank/lora-face-restoration/tree/main/Presentation_Slides)

Implemented LoRA personalization of the Gen2Res diffusion U-Net and compared dataset sizes, scaling, and attention, residual, and timestep-embedding adaptation using PSNR, SSIM, and LPIPS.

In the reported configuration, attention-only LoRA reduced fine-tuning time by 46% and GPU memory usage by 47%, with comparable perceptual quality. Per-identity storage decreased from 527 MB full checkpoints to 200 KB adapters, used with the shared base model.

Selected research code is provided here; datasets, model checkpoints, and full experimental outputs are maintained separately on the research server.

Based on [Gen2Res](https://github.com/adobe-research/gen2res). Inherited code and its modifications are subject to the [Adobe Research License](Code/LICENSE).
