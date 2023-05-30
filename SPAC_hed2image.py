import argparse

from share import *
import config
import os
import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import pytorch_lightning as pl

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.hed import HEDdetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from PIL import Image
import random
class seg2image_batch(pl.LightningModule):
    def __init__(self, opt):
        super(seg2image_batch, self).__init__()
        self.input_folder = opt.input_folder
        self.output_folder = opt.output_folder
        self.batch_size = opt.batch_size
        data_transformer = {'test': transforms.Compose([
            transforms.ToTensor(),
        ])}
        self.imagefolder = ImageFolder(self.input_folder, transform=data_transformer['test'])

        self.dataset = DataLoader(dataset=self.imagefolder, batch_size=self.batch_size, shuffle=False)
        self.apply_hed = HEDdetector()
        model = create_model('./models/cldm_v15.yaml').cpu()
        model.load_state_dict(load_state_dict('./models/control_sd15_hed.pth', location='cuda'))
        self.model = model.cuda()
        self.ddim_sampler = DDIMSampler(model)
        self.a_prompt = opt.a_prompt
        self.n_prompt = opt.n_prompt
        self.num_samples = opt.num_samples
        self.image_resolution = opt.image_resolution
        self.detect_resolution = opt.detect_resolution
        self.detect_resolution_bg = 512
        self.ddim_steps = opt.ddim_steps
        self.guess_mode = opt.guess_mode
        self.strength = opt.strength
        self.scale = opt.scale
        self.seed = opt.seed
        self.eta = opt.eta
        self.prompt = opt.prompt
        self.background = opt.bg_folder

    def read_random_image(self, folder_path):
        image_filenames = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        random_image_filename = random.choice(image_filenames)
        random_image_path = os.path.join(folder_path, random_image_filename)
        image = Image.open(random_image_path)
        image = self.crop_square(image)
        image_array = np.array(image, dtype=np.uint8)
        return image_array

    def find_contours(self, binary_image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE):
        contours, hierarchy = cv2.findContours(binary_image, mode, method)
        return contours, hierarchy

    def create_mask(self, image, contours, thickness=2):
        mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask.fill(0)
        cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness)
        return mask

    def erode_mask(self, mask, kernel_size=3, iterations=1):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        eroded_mask = cv2.erode(mask, kernel, iterations=iterations)
        return eroded_mask

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
    def combine_images_without_interior(self, a, b):
        assert a.shape == b.shape, "Images must have the same shape."
        combined = np.copy(a)
        # blur = gaussian_blur(a)
        combined2 = np.copy(combined)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        a = cv2.erode(a, kernel, iterations=1)
        # find the contour
        contours, _ = cv2.findContours(a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # create a black mask
        mask = np.zeros_like(a, dtype=np.uint8)
        # implant the inner area of contour
        cv2.drawContours(mask, contours, -1, (1, 1, 1), thickness=cv2.FILLED)
        # switch
        combined2[(combined == 0) & (mask == 0)] = b[(combined == 0) & (mask == 0)]

        return combined2
        # cv2.imshow("Eroded Binary Image", combined2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return combined2

    def process(self, input_image):
        with torch.no_grad():
            time_of_day = ["dawn", "sunset", "slightly bright", "sunny","slightly dusk"]
            #animal_environment = ["green grassland",  "bush", "Savannah","mountain","farm","zoo"]
            random_time = random.choice(time_of_day)
            #random_environment = random.choice(animal_environment)
            #random_weather = random.choice(weather)
            prompt = f"zebra, {random_time}, high resolution, natrual"
            print(prompt)
            random_integer = random.randint(331845405, 2147483647)
            self.prompt = prompt
            self.seed = random_integer
            input_image = HWC3(input_image)
            detected_map = self.apply_hed(resize_image(input_image, self.detect_resolution))
            img_array = self.read_random_image(self.background)
            img_array = HWC3(img_array)
            detected_map2 = self.apply_hed(resize_image(img_array, self.detect_resolution))
            detected_map = self.combine_images_without_interior(detected_map, detected_map2)
            detected_map = HWC3(detected_map)

            img = resize_image(input_image, self.image_resolution)
            H, W, C = img.shape

            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(self.num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            if self.seed == -1:
                seed = random.randint(0, 65535)
            else:
                seed = self.seed
            seed = 2047483647
            seed_everything(seed)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            cond = {"c_concat": [control], "c_crossattn": [
                self.model.get_learned_conditioning([self.prompt + ', ' + self.a_prompt] * self.num_samples)]}
            un_cond = {"c_concat": None if self.guess_mode else [control],
                       "c_crossattn": [self.model.get_learned_conditioning([self.n_prompt] * self.num_samples)]}
            shape = (4, H // 8, W // 8)
            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=True)

            self.model.control_scales = [self.strength * (0.825 ** float(12 - i)) for i in
                                         range(13)] if self.guess_mode else ([self.strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            samples, intermediates = self.ddim_sampler.sample(self.ddim_steps, self.num_samples,
                                                              shape, cond, verbose=False, eta=self.eta,
                                                              unconditional_guidance_scale=self.scale,
                                                              unconditional_conditioning=un_cond)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            x_samples = self.model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0,
                                                                                                               255).astype(
                np.uint8)

            results = [x_samples[i] for i in range(self.num_samples)]

        return results
    def save_result(self):
        for id, input in enumerate(self.dataset):
            input = torch.squeeze(input[0]).numpy()*255 #C, H,W
            input = np.transpose(input, (1,2,0))
            input = input.astype(np.uint8)
            results = self.process(input)
            img_name = os.path.basename(self.imagefolder.imgs[id][0])
            print(img_name)
            for batchid, result in enumerate(results):
                img = Image.fromarray(result)
                img.save(os.path.join(self.output_folder, f"{img_name[:-4]}.jpg"))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='controlnet_seg2image_batchprocessing')
    parser.add_argument('--input_folder', type=str, default='test/')
    parser.add_argument('--output_folder', type=str, default='output/')
    parser.add_argument('--bg_folder', type=str, default='background')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')

    parser.add_argument('--prompt', type=str, default="")
    parser.add_argument('--num_samples', type=int, default=1, help='num_samples')
    parser.add_argument('--a_prompt', type=str,
                        default="best quality, extremely detailed")
    parser.add_argument('--n_prompt', type=str, default="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality")
    parser.add_argument('--image_resolution', type=int, default=768, help='image_resolution from 256 to 768')
    parser.add_argument('--detect_resolution', type=int, default=768, help='detect_resolution')
    parser.add_argument('--ddim_steps', type=int, default=40, help='steps from 1 to 100')
    parser.add_argument('--scale', type=float, default=9, help='Guidance Scale from 0.1 to 30.0')
    parser.add_argument('--seed', type=int, default=1847483647, help='seed from -1 to 2147483647')
    parser.add_argument('--strength', type=float, default=1, help='Control Strength from 0.0 to 2.0')
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--guess_mode', type=bool, default=False)
    args = parser.parse_args()
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    seg_model = seg2image_batch(args)
    seg_model.save_result()
