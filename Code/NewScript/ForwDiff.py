import os
import argparse
import numpy as np
import torch as th
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Define the noise schedule
def get_beta_schedule(schedule_name, num_steps):
    """Returns a noise schedule (betas)."""
    if schedule_name == "linear":
        beta_start = 0.0001
        beta_end = 0.02
        return np.linspace(beta_start, beta_end, num_steps, dtype=np.float64)
    elif schedule_name == "cosine":
        return np.cos((np.linspace(0, np.pi / 2, num_steps) + 0.008) / 1.008) ** 2
    else:
        raise NotImplementedError(f"Unknown schedule: {schedule_name}")

# Compute alpha and related terms
def compute_alphas(betas):
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    return alphas, alphas_cumprod

# Forward diffusion function
def forward_diffusion(image_tensor, t, alphas_cumprod):
    """Adds noise to an image at timestep t."""
    noise = th.randn_like(image_tensor)  # Generate Gaussian noise
    sqrt_alpha_cumprod_t = th.tensor(np.sqrt(alphas_cumprod[t]), dtype=th.float32)
    sqrt_one_minus_alpha_cumprod_t = th.tensor(np.sqrt(1 - alphas_cumprod[t]), dtype=th.float32)
    return sqrt_alpha_cumprod_t * image_tensor + sqrt_one_minus_alpha_cumprod_t * noise

# Load images from a directory
def load_images(image_dir, image_size=(128, 128)):
    """Loads images from a directory and converts them to tensors."""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),  # Convert image to tensor in range [0,1]
    ])
    image_tensors = []
    filenames = []
    
    for filename in os.listdir(image_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, filename)
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image) * 2 - 1  # Normalize to [-1, 1]
            image_tensors.append(image_tensor)
            filenames.append(filename)
    
    return th.stack(image_tensors), filenames  # Stack into batch tensor

# Display comparison of original vs. noisy images in a 3x3 grid
def display_comparison(originals, noisies, filenames):
    """Displays a 3x3 grid of original vs. noisy images."""
    fig, axes = plt.subplots(3, 6, figsize=(12, 6))  # 3 rows, 6 columns (alternating original & noisy)
    
    for i in range(9):  # First 9 images
        orig_np = ((originals[i] + 1) / 2 * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        noisy_np = ((noisies[i] + 1) / 2 * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        axes[i // 3, (i % 3) * 2].imshow(orig_np)  # Original
        axes[i // 3, (i % 3) * 2].set_title(f"Original\n{filenames[i]}")
        axes[i // 3, (i % 3) * 2].axis("off")

        axes[i // 3, (i % 3) * 2 + 1].imshow(noisy_np)  # Noisy
        axes[i // 3, (i % 3) * 2 + 1].set_title("Noisy")
        axes[i // 3, (i % 3) * 2 + 1].axis("off")
    
    plt.tight_layout()
    plt.show()

def apply_forward_diffusion(image_dir, output_dir, schedule, num_steps, noise_step):
    """Applies forward diffusion to images in a directory and saves results."""
    print(f"Using {schedule} noise schedule with {num_steps} diffusion steps.")
    
    betas = get_beta_schedule(schedule, num_steps)
    _, alphas_cumprod = compute_alphas(betas)

    images, filenames = load_images(image_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    noisy_images = []

    for i, image in enumerate(images):
        noisy_image = forward_diffusion(image, noise_step, alphas_cumprod)
        noisy_images.append(noisy_image)

        # Save the noisy image
        noisy_np = ((noisy_image + 1) / 2 * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        Image.fromarray(noisy_np).save(os.path.join(output_dir, f"noisy_{filenames[i]}"))

    # Display the first 9 comparisons
    display_comparison(images[:9], noisy_images[:9], filenames[:9])

def main():
    parser = argparse.ArgumentParser(description="Apply forward diffusion to images")
    
    parser.add_argument("--image_dir", type=str, default="imgs_align", help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default="noisy_images", help="Directory to save noisy images")
    parser.add_argument("--schedule", type=str, default="linear", choices=["linear", "cosine"], help="Noise schedule type")
    parser.add_argument("--num_steps", type=int, default=100, help="Total number of diffusion steps")
    parser.add_argument("--noise_step", type=int, default=50, help="Step at which noise is applied")

    args = parser.parse_args()

    apply_forward_diffusion(args.image_dir, args.output_dir, args.schedule, args.num_steps, args.noise_step)

# Run the script
if __name__ == "__main__":
    main()
