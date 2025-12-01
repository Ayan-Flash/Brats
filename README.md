# Brain Tumor Detection Web Interface

A responsive web interface for the brain tumor detection system using AI and deep learning.

## 🚀 Quick Start

### 1. Start the Flask API Server

```bash
cd F:\BRATS\CODE
python app.py
```

The server will start on `http://localhost:5000`

### 2. Open the Web Interface

Open your browser and navigate to:
```
http://localhost:5000
```

## 📁 Project Structure

```
CODE/
├── app.py              # Flask backend API
├── static/
│   ├── index.html      # Main UI page
│   ├── style.css       # Styling
│   └── script.js       # Frontend logic
├── uploads/            # Temporary upload folder (auto-created)
└── README.md           # This file
```

## 🎯 Features

- **Drag & Drop Upload**: Easy MRI image upload
- **Real-time Analysis**: Instant tumor detection and classification
- **Multiple Tumor Types**: Detects Glioma, Meningioma, Pituitary, etc.
- **Segmentation**: Tumor boundary detection
- **Feature Extraction**: Radiomic features and tumor characteristics
- **Survival Prediction**: Estimated survival based on age and tumor features
- **Clinical Recommendations**: AI-generated treatment suggestions
- **Responsive Design**: Works on desktop, tablet, and mobile

## 🔧 API Endpoints

### `GET /api/health`
Health check endpoint

### `POST /api/analyze`
Main analysis endpoint
- **Parameters**:
  - `image` (file): MRI scan image (PNG, JPG, JPEG, NII, DCM)
  - `age` (optional): Patient age in years
- **Returns**: JSON with classification, segmentation, features, and survival prediction

### `GET /api/models/status`
Check availability of trained models

## 📊 Supported Image Formats

- PNG (`.png`)
- JPEG (`.jpg`, `.jpeg`)
- NIfTI (`.nii`)
- DICOM (`.dcm`)

Maximum file size: **16MB**

## 🎨 UI Components

### Upload Section
- Drag & drop area
- File browser
- Image preview
- Patient age input

### Results Section
- **Classification Card**: Tumor type and confidence
- **Segmentation Card**: Tumor coverage and pixels
- **Features Card**: Tumor characteristics
- **Survival Prediction Card**: Estimated survival time
- **Recommendations**: Clinical suggestions

## ⚙️ Configuration

Edit `app.py` to modify:
- Upload folder location
- Allowed file extensions
- Maximum file size
- Model paths

## 🔒 Security Notes

- Files are automatically deleted after analysis
- CORS is enabled for development (restrict in production)
- Input validation for file types and sizes
- Secure filename handling

## 🐛 Troubleshooting

### Server won't start
```bash
# Check if port 5000 is in use
netstat -ano | findstr :5000

# Use a different port
python app.py --port 5001
```

### Models not found
```bash
# Check model status
curl http://localhost:5000/api/models/status
```

### Upload fails
- Check file size (max 16MB)
- Verify file format is supported
- Check browser console for errors

## 📝 Example Usage

### Using curl
```bash
curl -X POST http://localhost:5000/api/analyze \
  -F "image=@scan.png" \
  -F "age=55"
```

### Using Python
```python
import requests

with open('scan.png', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/analyze',
        files={'image': f},
        data={'age': 55}
    )
    result = response.json()
    print(result)
```

## 🎓 Academic Use

This system is designed for:
- Research purposes
- Educational demonstrations
- Medical AI development
- Clinical decision support systems

**⚠️ Important**: This is a research tool. Always consult with qualified medical professionals for diagnosis and treatment decisions.

## 📄 License

For research and educational purposes only.

## 🤝 Support

For issues or questions, please check the main tumor.py documentation.
