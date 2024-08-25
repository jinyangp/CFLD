import argparse
from datetime import datetime
from diffusers import DDPMScheduler
from defaults import pose_transfer_C as cfg
from pose_transfer_train import build_model
from models import UNet, VariationalAutoencoder
import torch
import os
import numpy as np
import pandas as pd
from pose_utils import (cords_to_map, draw_pose_from_cords,
                        load_pose_cords_from_strings)
import random
from PIL import Image
from torchvision import transforms
import copy

'''
    Out of the 25 key points,  we use 18 of them.
    
    Nose: Red (e.g., RGB: [255, 0, 0])
    Neck: Blue (e.g., RGB: [0, 0, 255])
    Right Shoulder: Green (e.g., RGB: [0, 255, 0])
    Right Elbow: Cyan (e.g., RGB: [0, 255, 255])
    Right Wrist: Magenta (e.g., RGB: [255, 0, 255])
    Left Shoulder: Yellow (e.g., RGB: [255, 255, 0])
    Left Elbow: Orange (e.g., RGB: [255, 165, 0])
    Left Wrist: Purple (e.g., RGB: [128, 0, 128])
    Right Hip: Pink (e.g., RGB: [255, 192, 203])
    Right Knee: Brown (e.g., RGB: [165, 42, 42])
    Right Ankle: Grey (e.g., RGB: [128, 128, 128])
    Left Hip: Light Blue (e.g., RGB: [173, 216, 230])
    Left Knee: Dark Green (e.g., RGB: [0, 128, 0])
    Left Ankle: Light Green (e.g., RGB: [144, 238, 144])
    Right Eye: Dark Red (e.g., RGB: [139, 0, 0])
    Left Eye: Light Coral (e.g., RGB: [240, 128, 128])
    Right Ear: Dark Orange (RGB: [255, 140, 0])
    Left Ear: Light Steel Blue (RGB: [176, 196, 222])
'''

# TODO: Check ordering of keypoints and ensure that it is the same
colour_to_keypoint = {
    (255, 0, 0): "Nose",
    (0, 0, 255): "Neck",
    (139, 0, 0): "Right Eye",
    (240, 128, 128): "Left Eye",
    (255, 140, 0): "Right Ear",
    (176, 196, 222): "Left Ear",
    (0, 255, 0): "Right Shoulder",
    (0, 255, 255): "Right Elbow",
    (255, 0, 255): "Right Wrist",
    (255, 255, 0): "Left Shoulder",
    (255, 165, 0): "Left Elbow",
    (128, 0, 128): "Left Wrist",
    (255, 192, 203): "Right Hip",
    (173, 216, 230): "Left Hip",
    (165, 42, 42): "Right Knee",
    (128, 128, 128): "Right Ankle",
    (0, 128, 0): "Left Knee",
    (144, 238, 144): "Left Ankle",
}

def map_to_nearest_color(pixel):
    min_distance = float('inf')
    nearest_color = None
    for color in colour_to_keypoint.keys():
        distance = np.sqrt(np.sum((np.array(color) - np.array(pixel)) ** 2))
        if distance < min_distance:
            min_distance = distance
            nearest_color = color
    return nearest_color

def approximate_pixel_vals(pil_img, colour_to_keypoint):

    width, height = pil_img.size[0], pil_img.size[1]
    pixel_values = list(pil_img.getdata())
    # Apply color mapping to each pixel
    mapped_pixel_values = [map_to_nearest_color(pixel) for pixel in pixel_values]
    mapped_pixel_nps = np.reshape(np.array(mapped_pixel_values), (height,width,3))
    return mapped_pixel_nps

def get_bodypart_coords(img_pil, colour_to_keypoint):
    
    img_np = approximate_pixel_vals(img_pil, colour_to_keypoint)

    keypoints = {keypoint_id: [] for keypoint_id in colour_to_keypoint.values()}
    
    for color, keypoint_id in colour_to_keypoint.items():
        color_mask = np.all(img_np == np.array(color), axis=-1)
        if np.any(color_mask):
            keypoints[keypoint_id] = np.argwhere(color_mask)
    return keypoints

def compute_keypoint_coordinates(keypoints):
    
    keypoint_coordinates = {}
    for keypoint_id, coords in keypoints.items():
        if len(coords) == 0:
            keypoint_coordinates[keypoint_id] = np.array([-1,-1])
        else:
            coords = sorted(coords, key=lambda coord: coord[1])
            keypoint_coordinates[keypoint_id] = coords[0]

    return keypoint_coordinates

def generate(pair_idx: int,
             output_dir: str):

    # Load models
    noise_scheduler = DDPMScheduler.from_pretrained("pretrained_models/scheduler/scheduler_config.json")
    vae = VariationalAutoencoder(pretrained_path="pretrained_models/vae").eval().requires_grad_(False).cuda()
    model = build_model(cfg).eval().requires_grad_(False).cuda()
    unet = UNet(cfg).eval().requires_grad_(False).cuda()
    print(model.load_state_dict(torch.load(os.path.join("checkpoints", "pytorch_model.bin"), map_location="cpu"), strict=False))
    print(unet.load_state_dict(torch.load(os.path.join("checkpoints", "pytorch_model_1.bin"), map_location="cpu"), strict=False))

    test_pairs = os.path.join(os.getcwd(), "fashion", "fashion-resize-pairs-test.csv")
    test_pairs = pd.read_csv(test_pairs)

    # Get source image, source image pose and target image pose
    # random_index = random.choice(range(len(test_pairs)))
    random_index = 0
    # TODO: the pose used in original repo has a pose map which has multiple channels and the first 3 channels being the image
    # need to figure out a good way to get the pose map so we can be more flexible in our approach
    img_from_path, pose_img_path = test_pairs.iloc[random_index]["from"], test_pairs.iloc[random_index]["to_pose"]
    
    img_from = Image.open(os.path.join("fashion", "test_highres", img_from_path)).convert("RGB")
    img_from.resize((256,256))
    trans = transforms.Compose([
    transforms.Resize([256, 256], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    img_from_tensor = trans(img_from).unsqueeze(0)
    print(img_from_tensor.shape)

    pose_img_pil = Image.open(os.path.join("fashion", "test_highres", pose_img_path))
    pose_img_pil = pose_img_pil.resize((256,256))
    pose_img_np = np.array(pose_img_pil)

    # load in the pose_img_pil
    # STEP: Get the pose map
    bodypart_coords = get_bodypart_coords(pose_img_pil, colour_to_keypoint)
    keypoint_coords = compute_keypoint_coordinates(bodypart_coords)
    y_cords = np.array([int(coord[0]) for _,coord in keypoint_coords.items()])
    x_cords = np.array([int(coord[1]) for _,coord in keypoint_coords.items()])
    cords = np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)
    pose_map = torch.tensor(cords_to_map(cords, (256,256), (256, 256)).transpose(2,0,1), dtype=torch.float32)

    # STEP: Get the pose image
    pose_img = torch.tensor(pose_img_np.transpose(2, 0, 1) / 255., dtype=torch.float32)

    # STEP: Concatenate them along first dimension
    pose_img_tensor = torch.cat([pose_img, pose_map], dim=0).unsqueeze(0)
    print(pose_img_tensor.shape)

    with torch.no_grad():
        c_new, down_block_additional_residuals, up_block_additional_residuals = model({
            "img_cond": img_from_tensor.cuda(), "pose_img": pose_img_tensor.cuda()})
        noisy_latents = torch.randn((1, 4, 64, 64)).cuda()
        weight_dtype = torch.float32
        bsz = 1

        c_new = torch.cat([c_new[:bsz], c_new[:bsz], c_new[bsz:]])
        down_block_additional_residuals = [torch.cat([torch.zeros_like(sample), sample, sample]).to(dtype=weight_dtype) \
                                            for sample in down_block_additional_residuals]
        up_block_additional_residuals = {k: torch.cat([torch.zeros_like(v), torch.zeros_like(v), v]).to(dtype=weight_dtype) \
                                            for k, v in up_block_additional_residuals.items()}
 
        noise_scheduler.set_timesteps(cfg.TEST.NUM_INFERENCE_STEPS)
        for t in noise_scheduler.timesteps:
            inputs = torch.cat([noisy_latents, noisy_latents, noisy_latents], dim=0)
            inputs = noise_scheduler.scale_model_input(inputs, timestep=t)
            noise_pred = unet(sample=inputs, timestep=t, encoder_hidden_states=c_new,
                down_block_additional_residuals=copy.deepcopy(down_block_additional_residuals),
                up_block_additional_residuals=copy.deepcopy(up_block_additional_residuals))

            noise_pred_uc, noise_pred_down, noise_pred_full = noise_pred.chunk(3)
            noise_pred = noise_pred_uc + \
                            cfg.TEST.DOWN_BLOCK_GUIDANCE_SCALE * (noise_pred_down - noise_pred_uc) + \
                            cfg.TEST.FULL_GUIDANCE_SCALE * (noise_pred_full - noise_pred_down)
            noisy_latents = noise_scheduler.step(noise_pred, t, noisy_latents)[0]

        sampling_imgs = vae.decode(noisy_latents) * 0.5 + 0.5 # denormalize
        sampling_imgs = sampling_imgs.clamp(0, 1)

    if output_dir:
        img_filename = os.path.basename(img_from_path).split(".")[0]
        output_dir = os.path.join(os.getcwd(), output_dir)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        output_full_fp = os.path.join(os.getcwd(), output_dir, f'{img_filename}_generated_output.jpg')
        Image.fromarray((sampling_imgs[0] * 255.).permute((1, 2, 0)).long().cpu().numpy().astype(np.uint8)).resize((256, 256)).save(output_full_fp)
        print(f'Output saved at {output_full_fp}')

if __name__ == "__main__":

        # Get the current timestamp
    current_timestamp = datetime.now()
    # Format the timestamp in DDMMYY-HH:MM:SS format
    formatted_timestamp = current_timestamp.strftime("%d%m%y-%H:%M:%S")

    parser = argparse.ArgumentParser(description="Generate image using pre-trained models pipelines.")
    parser.add_argument("pair_idx", type=str, help="Pair's index in csv file.")
    parser.add_argument("output_dir", type=str, default=f"fashion/output/{formatted_timestamp}", help="Path to output directory.")

    args = parser.parse_args()

    generate(args.pair_idx,
             args.output_dir)