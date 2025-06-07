import os
import torch
import yaml
from PIL import Image
import torchvision
from torchvision import transforms
from src.models import load_model
from src.utils import InputPadder
from src.transport import create_transport, Sampler

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Image loader
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)  # Shape: [1, 3, H, W]

# Save image to path
def save_image(tensor, path):
    torchvision.utils.save_image(tensor, path)

# Core interpolation function
def interpolate(eden, sample_fn, frame0, frame1, args):
    h, w = frame0.shape[2:]
    image_size = [h, w]
    padder = InputPadder(image_size)

    difference = ((torch.mean(torch.cosine_similarity(frame0, frame1), dim=[1, 2]) - args["cos_sim_mean"]) / args["cos_sim_std"]).unsqueeze(1).to(device)

    cond_frames = padder.pad(torch.cat((frame0, frame1), dim=0))
    new_h, new_w = cond_frames.shape[2:]
    noise = torch.randn([1, new_h // 32 * new_w // 32, args["model_args"]["latent_dim"]]).to(device)

    denoise_kwargs = {"cond_frames": cond_frames, "difference": difference}
    samples = sample_fn(noise, eden.denoise, **denoise_kwargs)[-1]
    denoise_latents = samples / args["vae_scaler"] + args["vae_shift"]
    generated_frame = eden.decode(denoise_latents)
    generated_frame = padder.unpad(generated_frame.clamp(0., 1.))
    return generated_frame

# Entry point for Replicate
def run(
    frame_0_path: str,
    frame_1_path: str,
    config_path: str = "configs/eval_eden.yaml"
) -> str:
    # Load config
    with open(config_path, "r") as f:
        update_args = yaml.unsafe_load(f)

    args = update_args
    model_name = args["model_name"]

    # Load EDEN model
    eden = load_model(model_name, **args["model_args"])
    ckpt = torch.load(args["pretrained_eden_path"], map_location=device)
    eden.load_state_dict(ckpt["eden"])
    eden.to(device)
    eden.eval()
    del ckpt

    # Sampler
    transport = create_transport("Linear", "velocity")
    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(sampling_method="euler", num_steps=2, atol=1e-6, rtol=1e-3)

    # Load frames
    frame_0 = load_image(frame_0_path).to(device)
    frame_1 = load_image(frame_1_path).to(device)

    # Interpolate
    with torch.no_grad():
        interpolated_frame = interpolate(eden, sample_fn, frame_0, frame_1, args)

    # Save output
    output_path = "/outputs/interpolated.png"
    os.makedirs("/outputs", exist_ok=True)
    save_image(interpolated_frame, output_path)
    return output_path

