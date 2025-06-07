import argparse
import os
import torch
from PIL import Image
from src.model import EDEN  # Adjust the import path based on your project structure
from src.utils import load_image, save_image  # Ensure these utility functions are defined

def interpolate_frames(model, frame1, frame2):
    """
    Interpolates between two frames using the EDEN model.
    """
    with torch.no_grad():
        interpolated = model(frame1, frame2)
    return interpolated

def main():
    parser = argparse.ArgumentParser(description="EDEN Frame Interpolation")
    parser.add_argument("--frame1", type=str, required=True, help="Path to the first input frame.")
    parser.add_argument("--frame2", type=str, required=True, help="Path to the second input frame.")
    args = parser.parse_args()

    # Load input frames
    frame1 = load_image(args.frame1)
    frame2 = load_image(args.frame2)

    # Initialize and load the EDEN model
    model = EDEN()
    model.load_state_dict(torch.load("path_to_pretrained_model.pth", map_location="cpu"))
    model.eval()

    # Perform interpolation
    interpolated_frame = interpolate_frames(model, frame1, frame2)

    # Save the output
    output_path = "/outputs/interpolated_frame.png"
    save_image(interpolated_frame, output_path)
    print(f"Interpolated frame saved to {output_path}")

if __name__ == "__main__":
    main()
