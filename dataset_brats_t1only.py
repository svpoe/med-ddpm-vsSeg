#-*- coding:utf-8 -*-
# Modified BRATS dataset for T1-only generation
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Lambda
from glob import glob
from utils.dtypes_vs import LabelEnum
import matplotlib.pyplot as plt
import nibabel as nib
import torchio as tio
import numpy as np
import torch
import re
import os


class NiftiPairImageGeneratorT1Only(Dataset):
    """
    Modified BRATS dataset that uses BRATS processing but outputs only T1 modality
    """
    def __init__(self,
            input_folder: str,
            t1_folder: str,    # Only T1 folder needed
            input_size: int,
            depth_size: int,
            input_channel: int = 8,
            transform=None,
            target_transform=None,
            full_channel_mask=False,
            combine_output=False
        ):
        self.input_folder = input_folder
        self.t1_folder = t1_folder
        self.pair_files = self.pair_file()
        self.input_size = input_size
        self.depth_size = depth_size
        self.input_channel = input_channel
        self.transform = transform
        self.target_transform = target_transform
        self.full_channel_mask = full_channel_mask
        self.combine_output = combine_output

    def pair_file(self):
        input_files = sorted(glob(os.path.join(self.input_folder, '*')))
        target_files = sorted(glob(os.path.join(self.t1_folder, '*')))
        pairs = []
        for input_file, target_file in zip(input_files, target_files):
            print("input: ", input_file, ", target: ", target_file)
            # Simple filename matching for VS data
            try:
                input_num = int("".join(re.findall(r"\d+", os.path.basename(input_file))))
                target_num = int("".join(re.findall(r"\d+", os.path.basename(target_file))))
                assert input_num == target_num
                pairs.append((input_file, target_file))
            except (ValueError, AssertionError):
                print(f"Warning: Could not match {input_file} with {target_file}")
                continue
        return pairs

    def masks2label(self, masked_img):
        # Use BRATS label processing but adapted for VS data
        result_img = masked_img.copy().astype(np.float32)
        # Normalize label values for BRATS-style processing
        unique_vals = np.unique(result_img)
        if len(unique_vals) <= 2:  # Binary mask
            result_img[result_img > 0] = 1.0
        else:  # Multi-class mask
            result_img = result_img / np.max(result_img) if np.max(result_img) > 0 else result_img
        return result_img

    def label2masks(self, masked_img):
        # Convert labels to 3-channel mask format for VS data (background, brain, tumor)
        result_img = np.zeros(masked_img.shape + (3,))  # Only 3 channels for VS data
        unique_vals = np.unique(masked_img)
        
        if len(unique_vals) <= 2:  # Binary mask (background + tumor)
            result_img[masked_img > 0, 0] = 1  # Assign tumor to first channel
        else:  # Multi-class (background, brain, tumor)
            sorted_vals = sorted(unique_vals)[1:]  # Skip background (0)
            for i, val in enumerate(sorted_vals):
                if i < 3:  # Limit to 3 channels
                    result_img[masked_img == val, i] = 1
        return result_img

    def read_image(self, file_path):
        img = nib.load(file_path).get_fdata()
        return img

    def resize_img(self, img):
        h, w, d = img.shape
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            img = tio.ScalarImage(tensor=img[np.newaxis, ...])
            cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
            img = np.asarray(cop(img))[0]
        return img

    def resize_img_4d(self, input_img):
        h, w, d, c = input_img.shape
        result_img = np.zeros((self.input_size, self.input_size, self.depth_size, c))
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            for ch in range(c):
                buff = input_img.copy()[..., ch]
                img = tio.ScalarImage(tensor=buff[np.newaxis, ...])
                cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
                img = np.asarray(cop(img))[0]
                result_img[..., ch] += img
            return result_img
        else:
            return input_img

    def sample_conditions(self, batch_size: int):
        indexes = np.random.randint(0, len(self), batch_size)
        input_files = [self.pair_files[index][0] for index in indexes]
        input_tensors = []
        for input_file in input_files:
            input_img = self.read_image(input_file)
            input_img = self.label2masks(input_img) if self.full_channel_mask else input_img
            input_img = self.resize_img(input_img) if not self.full_channel_mask else self.resize_img_4d(input_img)
            if self.transform is not None:
                input_img = self.transform(input_img).unsqueeze(0)
                input_tensors.append(input_img)
        return torch.cat(input_tensors, 0).cuda()

    def __len__(self):
        return len(self.pair_files)

    def __getitem__(self, index):
        input_file, target_file = self.pair_files[index]
        
        # Process input (mask/segmentation)
        input_img = self.read_image(input_file)
        input_img = self.label2masks(input_img) if self.full_channel_mask else input_img
        input_img = self.resize_img(input_img) if not self.full_channel_mask else self.resize_img_4d(input_img)

        # Process target (T1 only)
        target_img = self.read_image(target_file)
        target_img = self.resize_img(target_img)
        
        # Apply transforms
        if self.transform is not None:
            input_img = self.transform(input_img)
        if self.target_transform is not None:
            target_img = self.target_transform(target_img)

        if self.combine_output:
            return torch.cat([target_img, input_img], 0)

        return {'input': input_img, 'target': target_img}
