import argparse
import os
import sys

import torch
torch.cuda.empty_cache()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..")))

import math
import time

import cv2
import numpy as np
import torch as th
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import create_dataloader
from guided_diffusion.script_util import (NUM_CLASSES, add_dict_to_argparser,
                                          args_to_dict,
                                          create_model_and_diffusion,
                                          model_and_diffusion_defaults)
from guided_diffusion.train_util import patch_checkpoint_for_lora


def img2tensor(imgs, bgr2rgb=True, float32=True):

  def _totensor(img, bgr2rgb, float32):
    if img.shape[2] == 3 and bgr2rgb:
      if img.dtype == 'float64':
        img = img.astype('float32')
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = th.from_numpy(img.transpose(2, 0, 1))
    if float32:
      img = img.float()
    return img

  if isinstance(imgs, list):
    return [_totensor(img, bgr2rgb, float32) for img in imgs]
  else:
    return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
  """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
  if not (th.is_tensor(tensor) or
          (isinstance(tensor, list) and all(th.is_tensor(t) for t in tensor))):
    raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

  if th.is_tensor(tensor):
    tensor = [tensor]
  result = []
  for _tensor in tensor:
    _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
    _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

    n_dim = _tensor.dim()
    if n_dim == 4:
      img_np = make_grid(_tensor,
                         nrow=int(math.sqrt(_tensor.size(0))),
                         normalize=False).numpy()
      img_np = img_np.transpose(1, 2, 0)
      if rgb2bgr:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    elif n_dim == 3:
      img_np = _tensor.numpy()
      img_np = img_np.transpose(1, 2, 0)
      if img_np.shape[2] == 1:  # gray image
        img_np = np.squeeze(img_np, axis=2)
      else:
        if rgb2bgr:
          img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    elif n_dim == 2:
      img_np = _tensor.numpy()
    else:
      raise TypeError('Only support 4D, 3D or 2D tensor. '
                      f'But received with dimension: {n_dim}')
    if out_type == np.uint8:
      # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
      img_np = (img_np * 255.0).round()
    img_np = img_np.astype(out_type)
    result.append(img_np)
  if len(result) == 1:
    result = result[0]
  return result


def main():
  args = create_argparser().parse_args()

  dist_util.setup_dist()
  # logger.configure()

  start_time = time.time()

  logger.log("creating data loader...")
  dataloader = create_dataloader(
      data_dir=args.data_dir,
      batch_size=args.batch_size,
      image_size=args.image_size,
      class_cond=args.class_cond,
  )

  end_time = time.time()
  print('create dataloader:' + str(end_time - start_time))

  logger.log("creating model and diffusion...")
  start_time = time.time()
  model, diffusion = create_model_and_diffusion(
      **args_to_dict(args,
                     model_and_diffusion_defaults().keys()), use_lora=args.use_lora)
  model.to(dist_util.dev())
  end_time = time.time()
  print('create model:' + str(end_time - start_time))

  start_time = time.time()
  # model.load_state_dict(dist_util.load_state_dict(args.model_path))
  base_state = dist_util.load_state_dict(args.model_path, map_location=dist_util.dev())
  base_state = patch_checkpoint_for_lora(base_state, model)
  model.load_state_dict(base_state, strict=False)

########################################################################
  if args.lora_path:
    logger.log(f"Loading LoRA weights from {args.lora_path}...")
    lora_state = torch.load(args.lora_path, map_location=dist_util.dev())
    filtered_state = {k: v for k, v in lora_state.items() if "lora_" in k}
    model.load_state_dict(filtered_state, strict=False)
#########################################################################
    for name, module in model.named_modules():
        if hasattr(module, "scaling"):
            old_scale = module.scaling
            module.scaling = args.lora_scale
            print(f"Adjusted LoRA scale for {name}: {old_scale:.4f} → {module.scaling:.4f}")
    for name, module in model.named_modules():
        if hasattr(module, "lora_down") and hasattr(module.lora_down, "weight"):
            norm = module.lora_down.weight.norm().item()
            print(f"{name}.lora_down.weight norm: {norm:.6f}")
    print("Final scaling check:")
    
    for name, module in model.named_modules():
        if hasattr(module, "scaling"):
            print(f"{name}: scaling = {module.scaling}")

####################################################################

  end_time = time.time()
  print('load model:' + str(end_time - start_time))

  if args.use_fp16:
    model.convert_to_fp16()
  model.eval()

  logger.log("sampling...")

  os.makedirs(args.outputdir, exist_ok=True)
  tot = 0
  TOT = len(dataloader)

  assert args.batch_size == 1, 'batch size must be 1'

  for data, dic in dataloader:

    data = data.to(dist_util.dev())
    model_kwargs = {}

    sample_fn = (diffusion.p_sample_loop
                 if not args.use_ddim else diffusion.ddim_sample_loop)
    sample = sample_fn(
        model,
        (data.shape[0], 3, args.image_size, args.image_size),
        clip_denoised=args.clip_denoised,
        model_kwargs=model_kwargs,
        ref_img=data if data.shape[-1] == args.image_size else F.interpolate(
            data, (args.image_size, args.image_size)),
        last=args.last,
        skip=args.skip,
        scale=args.scale,
        noise_step=args.noise_step,
        first=args.first,
    )

    save_image((sample[0:1] + 1) / 2., os.path.join(args.outputdir,
                                                    dic['filename'][0]))

    if args.num != -1 and tot >= args.num:
      break


def create_argparser():
  defaults = dict(
      clip_denoised=True,
      batch_size=1,
      use_ddim=False,
      model_path="",
      outputdir="",
      data_dir="",
      last=200,
      skip=10,
      scale=0.3,
      noise_step=0,
      first=0,
      mode="",
      num=-1,
      lora_path="",
      use_lora=False,
      lora_scale=1.0,
  )
  defaults.update(model_and_diffusion_defaults())
  parser = argparse.ArgumentParser()
  add_dict_to_argparser(parser, defaults)
  return parser


if __name__ == "__main__":
  main()
