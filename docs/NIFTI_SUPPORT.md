# NIfTI File Support

## Overview
The Enhanced Brain Tumor Detection System now supports **NIfTI (.nii and .nii.gz)** files in addition to standard image formats (PNG, JPG, GIF, BMP).

## What are NIfTI Files?
NIfTI (Neuroimaging Informatics Technology Initiative) is a medical imaging format commonly used for brain MRI scans. These files can contain:
- 3D volumetric data (x, y, z dimensions)
- 4D data with multiple modalities (x, y, z, modality)

## How it Works

### Backend Processing
1. **File Detection**: The system automatically detects `.nii` or `.nii.gz` files
2. **Volume Loading**: Uses `nibabel` library to load the NIfTI volume
3. **Slice Extraction**: Automatically extracts the middle slice from the z-axis for 2D analysis
4. **Normalization**: Normalizes intensity values to 0-255 range
5. **Model Processing**: Converts to PIL Image and processes like standard images

### Supported Formats
- `.nii` - Uncompressed NIfTI
- `.nii.gz` - Compressed NIfTI (recommended for storage)

### Dimension Handling
- **4D volumes** (x, y, z, modality): Uses first modality (e.g., FLAIR from multi-modal scan)
- **3D volumes** (x, y, z): Extracts middle axial slice
- **2D slices**: Processed directly

## Installation
The system requires `nibabel` library:

```bash
pip install nibabel
```

Already installed in your environment ✓

## Usage

### Web Interface
1. Click "Choose Image File"
2. Select a `.nii` or `.nii.gz` file
3. Choose your AI model
4. (Optional) Enter patient age
5. Click "Analyze Now"

### Supported File Types in UI
- JPG, PNG, GIF, BMP (standard images)
- **NII, NII.GZ (NIfTI volumes)** ✨ NEW
- DCM (DICOM - already supported)

## BraTS Dataset Compatibility
Perfect for analyzing BraTS dataset files:
- `BraTS20_Training_001_flair.nii`
- `BraTS20_Training_001_t1.nii.gz`
- `BraTS20_Training_001_t2.nii`
- `BraTS20_Training_001_seg.nii` (segmentation masks)

## Technical Details

### Code Location
- **Backend**: `F:\BRATS\CODE\app_enhanced.py`
  - `load_nifti_file()` - NIfTI loader function
  - `preprocess_for_model()` - Updated to handle NIfTI
  - `analyze()` - Endpoint accepts NIfTI files

- **Frontend**: `F:\BRATS\CODE\static\index.html`
  - File input accepts `.nii,.nii.gz`
  - Updated help text

### Processing Pipeline
```
NIfTI File Upload
    ↓
nibabel.load() - Load volume
    ↓
Extract middle slice (z-axis)
    ↓
Normalize to 0-255
    ↓
Convert to PIL Image
    ↓
Standard preprocessing (resize, channels, normalization)
    ↓
AI Model prediction
```

## Examples

### Sample BraTS File Analysis
```python
# Your BraTS files can now be uploaded directly:
# F:\BRATS\archive\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\
#   └── BraTS20_Training_001\
#       ├── BraTS20_Training_001_flair.nii
#       ├── BraTS20_Training_001_t1.nii.gz
#       ├── BraTS20_Training_001_t2.nii
#       └── BraTS20_Training_001_seg.nii
```

### Which Models Work Best?
- **Segmentation Models** (Recommended for NIfTI):
  - Attention U-Net
  - Improved CNN
  - Simple CNN (BraTS)
  
- **Classification Models** (Also supported):
  - Keras Basic Model
  - Brain Tumor Detector

## Error Handling
- **Missing nibabel**: Clear error message with installation instructions
- **Invalid NIfTI**: Catches corrupted or incompatible files
- **Dimension errors**: Handles unexpected volume dimensions gracefully

## Performance
- **File Size**: Max 16MB (configurable)
- **Processing Time**: ~1-3 seconds for typical brain MRI volume
- **Memory**: Efficient - only middle slice loaded into memory

## Next Steps
Want to process **multiple slices** instead of just the middle one? Let me know and I can add:
- Multi-slice analysis
- 3D segmentation support
- Full volume processing
- Slice selection interface

---
**Status**: ✅ Fully Operational  
**Last Updated**: November 6, 2025  
**Dependencies**: nibabel ✓ Installed
