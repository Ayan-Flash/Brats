import os
import numpy as np
import nibabel as nib
from PIL import Image
import matplotlib.pyplot as plt

def extract_test_images():
    base_path = r"F:\BRATS\CODE\archive\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001"
    t2_path = os.path.join(base_path, "BraTS20_Training_001_t2.nii")
    seg_path = os.path.join(base_path, "BraTS20_Training_001_seg.nii")
    
    if not os.path.exists(t2_path) or not os.path.exists(seg_path):
        print("Could not find NIfTI files.")
        return

    print("Loading NIfTI files...")
    t2_img = nib.load(t2_path).get_fdata()
    seg_img = nib.load(seg_path).get_fdata()
    
    # Find slice with largest tumor area
    print("Finding slice with largest tumor...")
    max_tumor_area = 0
    tumor_slice_idx = -1
    
    for i in range(seg_img.shape[2]):
        area = np.sum(seg_img[:, :, i] > 0)
        if area > max_tumor_area:
            max_tumor_area = area
            tumor_slice_idx = i
            
    print(f"Largest tumor found at slice {tumor_slice_idx} (Area: {max_tumor_area})")
    
    # Find a slice with NO tumor (but with brain)
    print("Finding non-tumor slice...")
    no_tumor_slice_idx = -1
    center_slice = seg_img.shape[2] // 2
    
    # Search outwards from center to find a slice with brain but no tumor
    for i in range(seg_img.shape[2]):
        # Check if slice has brain tissue (non-zero in T2) but no tumor (zero in Seg)
        has_brain = np.sum(t2_img[:, :, i] > 10) > 1000
        has_tumor = np.sum(seg_img[:, :, i] > 0) > 0
        
        if has_brain and not has_tumor:
            no_tumor_slice_idx = i
            break
            
    print(f"Non-tumor slice found at {no_tumor_slice_idx}")
    
    # Extract and save
    os.makedirs('test_samples', exist_ok=True)
    
    def save_slice(slice_data, filename):
        # Normalize to 0-255
        normalized = ((slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data)) * 255).astype(np.uint8)
        img = Image.fromarray(normalized)
        img.save(filename)
        print(f"Saved {filename}")

    if tumor_slice_idx != -1:
        save_slice(t2_img[:, :, tumor_slice_idx], 'test_samples/real_tumor.png')
        
    if no_tumor_slice_idx != -1:
        save_slice(t2_img[:, :, no_tumor_slice_idx], 'test_samples/real_no_tumor.png')

if __name__ == "__main__":
    extract_test_images()
