import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import transform
import nibabel as nib  # For loading MRI images
import matplotlib.pyplot as plt

np.random.seed(5)

def load_nii_volume(file_path):
    """Load and return a 3D NII volume as a tensor, cast to float32."""
    nii_img = nib.load(file_path)
    volume = torch.tensor(nii_img.get_fdata(), dtype=torch.float32)  # Convert to a float32 tensor
    return volume

def slice_nii_volume_to_2d(volume):
    """Slice a 3D NII volume into a list of 2D slices with debugging info."""
    slices = []
    for i in range(volume.shape[2]):  # Assuming shape is (H, W, D)
        slice_2d = volume[:, :, i]  # Extract 2D slice along the depth dimension
        
        # Convert to NumPy array for skimage processing
        slice_2d_np = slice_2d.numpy()
        
        # Skip slices that are empty or have very low variance
        if slice_2d_np.max() - slice_2d_np.min() < 1e-5:
            print(f"Skipping slice {i} due to low variance")
            continue
        
        # Resize to 64x64 as expected by the model
        slice_2d_resized = transform.resize(slice_2d_np, (64, 64), mode="constant")
        
        # Convert back to a tensor and add channel dimension
        slice_2d_tensor = torch.unsqueeze(torch.tensor(slice_2d_resized, dtype=torch.float32), 0)  # (1, H, W)
        slices.append(slice_2d_tensor)
    return slices

class MRIPairedDataset(Dataset):
    """
    Dataset loader for paired anatomical and fat fraction MRI images,
    where each 2D slice from the 3D volume is treated as a separate item.
    """
    def __init__(self, anatomical_folder, fat_fraction_folder):
        self.anatomical_slices = []
        self.fat_fraction_slices = []

        # Load and slice each 3D NIfTI volume
        anatomical_files = sorted([os.path.join(anatomical_folder, f) for f in os.listdir(anatomical_folder) if f.endswith('.nii')])
        fat_fraction_files = sorted([os.path.join(fat_fraction_folder, f) for f in os.listdir(fat_fraction_folder) if f.endswith('.nii')])

        # Ensure equal number of paired images in both folders
        assert len(anatomical_files) == len(fat_fraction_files), "Mismatched number of anatomical and fat fraction files"

        for anat_file, fat_file in zip(anatomical_files, fat_fraction_files):
            # Load and slice each volume, then extend to the list to flatten slices
            anatomical_volume = load_nii_volume(anat_file)
            fat_fraction_volume = load_nii_volume(fat_file)

            anatomical_slices = slice_nii_volume_to_2d(anatomical_volume)
            fat_fraction_slices = slice_nii_volume_to_2d(fat_fraction_volume)

            # Ensure we only add matching slice counts
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

if __name__ == "__main__":
    dataset = MRIPairedDataset("anatomical_folder_path", "fat_fraction_folder_path")
    gen = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=1)  # Set batch_size to any preferred value

    for i, (anatomical_slices, fat_fraction_slices) in enumerate(gen):
        print(f"Batch {i}: anatomical slices shape {anatomical_slices.shape}, fat fraction slices shape {fat_fraction_slices.shape}")
        
        # Display the first anatomical and fat fraction slices from the batch
        anatomical_img = anatomical_slices[0].squeeze().numpy()  # Select first slice and squeeze out singleton dimensions
        fat_fraction_img = fat_fraction_slices[0].squeeze().numpy()

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow((anatomical_img + 1) / 2, cmap="gray")  # Scale to [0, 1] for display
        plt.title("Anatomical MRI")

        plt.subplot(1, 2, 2)
        plt.imshow((fat_fraction_img + 1) / 2, cmap="gray")
        plt.title("Fat Fraction MRI")

        plt.show()
        break  # Displaying only the first batch
