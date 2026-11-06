# Code guide

This folder contains the research code accompanying the [project report](../report.pdf). It is organized to show the model adaptations and experiment drivers. It is not a validated, standalone software release.

## Organization

| Folder | Contents |
| --- | --- |
| `guided_diffusion/` | Diffusion model, LoRA layers, model factories, and training utilities |
| `scripts/` | Main experiment drivers, with descriptive names |
| `tools/` | Additional face-alignment and forward-diffusion utilities |
| `archive/` | Files originally supplied with an `old` prefix, retained for reference |

## Experiment drivers

The labels below describe the intended experiments represented by the scripts. Existing execution issues are listed separately below.

| Intended experiment | Training driver | Restoration driver |
| --- | --- | --- |
| Full fine-tuning baseline | [finetune_full.py](scripts/finetune_full.py) | [restore.py](scripts/restore.py) |
| Attention LoRA | [finetune_lora_attention.py](scripts/finetune_lora_attention.py) | [restore_lora_attention.py](scripts/restore_lora_attention.py) |
| Attention + residual-convolution LoRA | [finetune_lora_attention_residual.py](scripts/finetune_lora_attention_residual.py) | [restore_lora_attention_residual.py](scripts/restore_lora_attention_residual.py) |

[train_base.py](scripts/train_base.py) contains the base-model training driver, and [align.py](scripts/align.py) contains the main alignment utility. The attention + timestep-embedding experiment is represented by `unetemb.py`, `loraemb.py`, `script_utilemb.py`, and `train_utilemb.py`; a dedicated, wired entry script for that variant was not included in the uploaded source.

The original `1D` labels refer to the attention-projection adaptation. The `2D` variant adds residual-convolution adaptation; both variants operate on facial images.

## Model and adaptation code

- Attention adaptation: `lora.py`, `unet.py`, `script_util.py`, and `train_util.py`.
- Residual-convolution experiment: `lora2d.py`, `unet2d.py`, `script_util2d.py`, and `train_util2d.py`.
- Timestep-embedding experiment: `loraemb.py`, `unetemb.py`, `script_utilemb.py`, and `train_utilemb.py`.
- `train_utillpips.py` is retained as an additional experimental training utility.

## Additional tools

- [visualize_forward_diffusion.py](tools/visualize_forward_diffusion.py): generates noisy images and displays forward-diffusion comparisons.
- [align2.py](tools/align2.py): an additional face-alignment implementation.
- [newalign.py](tools/newalign.py): an alternative landmark-based face-alignment implementation. Its landmark-predictor file must be supplied separately.

## Data and model weights

Training images, test images, the pretrained base model, personalized checkpoints, and generated outputs are maintained outside this repository. The small per-identity LoRA adapters supplement the shared base model; they are not standalone restoration models. Refer to the report for the measured efficiency and image-quality results.

## Execution status

This cleanup preserves the uploaded research implementation. It does not reconstruct the original server environment or rerun the reported experiments. In particular, the following issues were already present in the source snapshot and remain documented rather than being resolved by guessing the experiment settings:

1. Both LoRA restoration drivers import `patch_checkpoint_for_lora` from `guided_diffusion.train_util`, where that helper is absent. Related helpers exist in other training-utility variants, but their intended use needs confirmation against the working experiment copy.
2. The residual training driver imports the default model factory while passing `lora_scale_att` and `lora_scale_conv`, which that factory does not accept. Its restoration driver also imports the default model factory.
3. The specialized `script_util2d.py` and `script_utilemb.py` factories still import `unet.py`; `unet2d.py` and `unetemb.py` import adapters from `lora.py` rather than their correspondingly named adapter modules. In particular, the active `lora.py` does not define the `LoRAConv2d` class imported by `unet2d.py`.
4. Environment versions, model checkpoints, launch commands, and per-experiment settings are not supplied in this snapshot. Some tools use working-directory-relative resource paths.

The published code was checked for Python syntax, file preservation, and the effects of file moves. These checks do not establish that training or restoration runs end to end.

## Attribution and license

The project builds on [Gen2Res: Restoration by Generation with Constrained Priors](https://github.com/adobe-research/gen2res), by Zheng Ding, Xuaner Zhang, Zhuowen Tu, and Zhihao Xia (CVPR 2024). The LoRA adaptations and comparative experiments are described in Sam Kang's report.

The upstream [Adobe Research License](LICENSE) is reproduced unchanged for the inherited code and its modifications. The report and slides are separate project materials; this copy of the code license does not assign a new license to them.

See the [cleanup notes](../docs/cleanup_notes.md) and [complete file map](../docs/file_map.csv) for the organization changes and verification details.
