import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib  # For loading MRI images
import matplotlib.pyplot as plt
import torch.nn.functional as F  # Import for padding

np.random.seed(5)

def load_nii_volume(file_path):
    nii_img = nib.load(file_path)
    volume = torch.tensor(nii_img.get_fdata(), dtype=torch.float32)
    return volume


def slice_nii_volume_to_2d(volume):
    """Slice a 3D NII volume into a list of 2D slices with padding to 64x64."""
    slices = []
    for i in range(volume.shape[2]):  # Assuming shape is (H, W, D)
        slice_2d = volume[:, :, i]  # Extract 2D slice along the depth dimension
        
        # Convert to NumPy array for skimage processing
        slice_2d_tensor = slice_2d.unsqueeze(0).unsqueeze(0)  # Make it (1, 1, H, W)
        
        # Padding to (64, 64)
        pad_h = (128 - slice_2d_tensor.shape[2]) // 2
        pad_w = (128- slice_2d_tensor.shape[3]) // 2
        slice_2d_padded = F.pad(slice_2d_tensor, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
        
        # Add padded slice to list and remove added dimensions
        slices.append(slice_2d_padded.squeeze(0))
    return slices


class MRIPairedDataset(Dataset):
    def __init__(self, anatomical_folder, fat_fraction_folder):
        self.anatomical_slices = []
        self.fat_fraction_slices = []

        anatomical_files = sorted([os.path.join(anatomical_folder, f) for f in os.listdir(anatomical_folder) if f.endswith('.nii')])
        fat_fraction_files = sorted([os.path.join(fat_fraction_folder, f) for f in os.listdir(fat_fraction_folder) if f.endswith('.nii')])

        assert len(anatomical_files) == len(fat_fraction_files), "Mismatched number of anatomical and fat fraction files"

        for anat_file, fat_file in zip(anatomical_files, fat_fraction_files):
            anatomical_volume = load_nii_volume(anat_file)
            fat_fraction_volume = load_nii_volume(fat_file)

            anatomical_slices = slice_nii_volume_to_2d(anatomical_volume)
            fat_fraction_slices = slice_nii_volume_to_2d(fat_fraction_volume)

            min_slices = min(len(anatomical_slices), len(fat_fraction_slices))
            self.anatomical_slices.extend(anatomical_slices[:min_slices])
            self.fat_fraction_slices.extend(fat_fraction_slices[:min_slices])

        self.len = len(self.anatomical_slices)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        anatomical_slice = self.anatomical_slices[idx]
        fat_fraction_slice = self.fat_fraction_slices[idx]
        return anatomical_slice, fat_fraction_slice

def visualize_and_save_sample(dataset, save_path="output_images/originaltest.png"):
    gen = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=1)

    for i, (anatomical_slices, fat_fraction_slices) in enumerate(gen):
        anatomical_img = anatomical_slices[0].squeeze().numpy()
        fat_fraction_img = fat_fraction_slices[0].squeeze().numpy()

        anatomical_img = (anatomical_img - anatomical_img.min()) / (anatomical_img.max() - anatomical_img.min())
        fat_fraction_img = (fat_fraction_img - fat_fraction_img.min()) / (fat_fraction_img.max() - fat_fraction_img.min())

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(anatomical_img, cmap="gray")
        plt.title("Anatomical MRI (Normalized)")

        plt.subplot(1, 2, 2)
        plt.imshow(fat_fraction_img, cmap="gray")
        plt.title("Fat Fraction MRI (Normalized)")

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.show()
        
        print(f"Saved image to: {save_path}")
        break
