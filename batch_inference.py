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

INPUT_WIDTH = 256
INPUT_HEIGHT = 256

def convert_fname(x):
    a, b = os.path.split(x)
    i = b.rfind('_')
    x = a + '/' +b[:i] + b[i+1:]
    return 'fashion'+x.split('.jpg')[0].replace('id_','id').replace('/','')

def get_name(src, dst):
    src = convert_fname(src)
    dst = convert_fname(dst)
    return src + '___' + dst

def resize_img(img: Image.Image,
               new_width: int,
               new_height: int) -> Image.Image:
    return img.resize((new_width, new_height))
    
def build_pose_img(annotation_file, img_path):
    string = annotation_file.loc[os.path.basename(img_path)]
    array = load_pose_cords_from_strings(string['keypoints_y'], string['keypoints_x'])
    pose_map = torch.tensor(cords_to_map(array, (256, 256), (256, 176)).transpose(2, 0, 1), dtype=torch.float32)
    pose_img = torch.tensor(draw_pose_from_cords(array, (256, 256), (256, 176)).transpose(2, 0, 1) / 255., dtype=torch.float32)
    pose_img = torch.cat([pose_img, pose_map], dim=0)
    return pose_img

def generate(data_csv_fp:str,
             project_name:str):
    
    # STEP: Load models
    noise_scheduler = DDPMScheduler.from_pretrained("pretrained_models/scheduler/scheduler_config.json")
    vae = VariationalAutoencoder(pretrained_path="pretrained_models/vae").eval().requires_grad_(False).cuda()
    model = build_model(cfg).eval().requires_grad_(False).cuda()
    unet = UNet(cfg).eval().requires_grad_(False).cuda()
    print(model.load_state_dict(torch.load(os.path.join("checkpoints", "pytorch_model.bin"), map_location="cpu"), strict=False))
    print(unet.load_state_dict(torch.load(os.path.join("checkpoints", "pytorch_model_1.bin"), map_location="cpu"), strict=False))

    # STEP: Load in data and annotation csv files
    test_pairs = os.path.join(os.getcwd(), "fashion", data_csv_fp)
    test_pairs = pd.read_csv(test_pairs)
    annotation_file = pd.read_csv(os.path.join("fashion", "fasion-resize-annotation-test.csv"), sep=':')
    annotation_file = annotation_file.set_index('name')
    image_root = os.path.join(os.getcwd(), "fashion")

    # STEP: Create output directories
    output_root_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(output_root_dir, exist_ok=True)
    output_dir = os.path.join(output_root_dir, project_name)
    samples_dir = os.path.join(output_dir, "samples")
    concat_dir = os.path.join(output_dir, "concat")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(concat_dir, exist_ok=True)
    f_ext = "png"
    
    for idx, row in test_pairs.iterrows():

        fname = get_name(row['from'], row['to'])
        src_img_fp = os.path.join(image_root, row['from'])
        target_img_fp = os.path.join(image_root, row['to'])
        
        src_img_pil = Image.open(src_img_fp)
        src_img_pil = resize_img(src_img_pil, INPUT_WIDTH, INPUT_HEIGHT)
        src_img_tensor = torch.tensor(np.array(src_img_pil)).permute(2,0,1)        

        target_img_pil = Image.open(target_img_fp)
        target_img_pil = resize_img(target_img_pil, INPUT_WIDTH, INPUT_HEIGHT)
        target_img_tensor = torch.tensor(np.array(target_img_pil)).permute(2,0,1)

        # STEP: Get source/style image
        img_from = Image.open(src_img_fp).convert("RGB")
        img_from.resize((256,256))
        
        trans = transforms.Compose([
        transforms.Resize([256, 256], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        
        # shape: torch.Size([1, 3, 256, 256])
        img_from_tensor = trans(img_from).unsqueeze(0)

        # STEP: Get the pose map
        converted_target_img_fp = convert_fname(row['to']) + ".jpg"
        pose_img_tensor = build_pose_img(annotation_file, converted_target_img_fp).unsqueeze(0)

        # STEP: Perform inference and generate image
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

        # STEP: Save the sample result
        Image.fromarray((sampling_imgs[0] * 255.).permute((1, 2, 0)).long().cpu().numpy().astype(np.uint8)).resize((256, 256)).save(os.path.join(samples_dir, f'{fname}.{f_ext}'))
        
        # STEP: Save the concat result
        sample_pil = Image.fromarray((sampling_imgs[0] * 255.).permute((1, 2, 0)).long().cpu().numpy().astype(np.uint8)).resize((256, 256))
        sample_pil = resize_img(sample_pil, INPUT_WIDTH, INPUT_HEIGHT)
        sample_tensor = torch.tensor(np.array(sample_pil)).permute(2,0,1)
        concat = transforms.Resize([256, 528])(torch.cat([src_img_tensor.detach().cpu(),
                                                     target_img_tensor.detach().cpu(),
                                                     sample_tensor.detach().cpu()], 2))
        transforms.ToPILImage()(concat).save(os.path.join(concat_dir, f'{fname}.{f_ext}'))

if __name__ == "__main__":

    # Get the current timestamp
    current_timestamp = datetime.now()
    # Format the timestamp in DDMMYY-HH:MM:SS format
    formatted_timestamp = current_timestamp.strftime("%d%m%y-%H:%M:%S")

    parser = argparse.ArgumentParser(description="Generate image using pre-trained models pipelines.")
    parser.add_argument("data_csv_fp", type=str, help="Path to data csv file containing from and to pairs.")
    parser.add_argument("project_name", type=str, default=f"Folder to save results under.")
    args = parser.parse_args()

    generate(args.data_csv_fp,
             args.project_name)