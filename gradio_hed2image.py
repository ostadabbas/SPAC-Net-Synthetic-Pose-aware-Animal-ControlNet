#the code is based on ControlNet https://github.com/lllyasviel/ControlNet
from share import *
import config
import os
import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.hed import HEDdetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from PIL import Image


def read_image_as_numpy_array(image_path):
    img = Image.open(image_path)
    img_array = np.array(img, dtype=np.uint8)
    return img_array
def read_random_image(folder_path):
    image_filenames = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    random_image_filename = random.choice(image_filenames)
    random_image_path = os.path.join(folder_path, random_image_filename)
    image = Image.open(random_image_path)
    image = crop_square(image)
    image_array = np.array(image, dtype=np.uint8)
    return image_array
def gaussian_blur(image, kernel_size=(9, 9), sigma_x=2, sigma_y=2):
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma_x, sigma_y)
    return blurred_image
def crop_square(self, img):
    width, height = img.size
    side_length = min(width, height)

    # Randomly choose the top-left corner coordinates for cropping
    left = random.randint(0, width - side_length)
    top = random.randint(0, height - side_length)

    # Compute the right and bottom coordinates for cropping
    right = left + side_length
    bottom = top + side_length

    # Crop the image to create a square
    square_image = img.crop((left, top, right, bottom))
    return square_image
def combine_images_without_interior(a, b):
    assert a.shape == b.shape, "Images must have the same shape."
    combined = np.copy(a)
    #blur = gaussian_blur(a)
    combined2 = np.copy(combined)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    a = cv2.erode(a, kernel, iterations=1)
    # find the contour
    contours, _ = cv2.findContours(a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # create a black mask
    mask = np.zeros_like(a, dtype=np.uint8)
    # implant the inner area of contour
    cv2.drawContours(mask, contours, -1, (1,1,1), thickness=cv2.FILLED)
    # switch
    combined2[(combined == 0) & (mask == 0)] = b[(combined == 0) & (mask == 0)]

    return combined2
apply_hed = HEDdetector()
image_path = '/mnt/u/ControlNet-main/ControlNet-main/test_imgs/bg2.jpg'
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_sd15_hed.pth', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        input_image = HWC3(input_image)
        detected_map = apply_hed(resize_image(input_image, detect_resolution))
        img_array = read_image_as_numpy_array(image_path)
        img_array = HWC3(img_array)
        detected_map2 = apply_hed(resize_image(img_array, detect_resolution))
        detected_map = combine_images_without_interior(detected_map,detected_map2)
        detected_map = np.array(detected_map, dtype=np.uint8)
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [detected_map] + results


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Control Stable Diffusion with HED Maps")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                detect_resolution = gr.Slider(label="HED Resolution", minimum=128, maximum=1024, value=512, step=1)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
    ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])


block.launch(server_name='0.0.0.0', share=True)
