"""
Enhanced Brain Tumor Detection API
Integrates all models and comprehensive feature extraction
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import io
import os
import sys
from datetime import datetime
from tensorflow.keras.models import load_model
import cv2

# Try to import nibabel for NIfTI support
try:
    import nibabel as nib
    NIFTI_SUPPORT = True
except ImportError:
    print("Warning: nibabel not installed. NIfTI (.nii) files will not be supported.")
    print("Install with: pip install nibabel")
    NIFTI_SUPPORT = False

# Add parent directory to import tumor.py functions
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from tumor import (
        validate_input_image,
        preprocess_mri_for_segmentation,
        run_segmentation_stage,
        extract_radiomic_features,
        predict_survival_from_features
    )
    ADVANCED_PIPELINE = True
except ImportError:
    print("Warning: Advanced tumor.py pipeline not available. Using basic classification only.")
    ADVANCED_PIPELINE = False

app = Flask(__name__, static_folder='static')
CORS(app)

# Configuration
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CODE_DIR)
UPLOAD_FOLDER = os.path.join(CODE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'nii', 'dcm'}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# All available models from both systems
# All available models from both systems
ALL_MODELS = {
    'best_segmentation': {
        'path': os.path.join(CODE_DIR, 'models', 'best_simple_cnn.h5'),
        'name': 'Best Segmentation (CNN)',
        'type': 'segmentation',
        'description': 'Top ranked BraTS trained segmentation model',
        'classes': {0: 'Background', 1: 'Tumor'},
        'input_size': (240, 240),
        'channels': 2
    },
    'keras_basic': {
        'path': os.path.join(CODE_DIR, 'models', 'my_model.keras'),
        'name': 'Keras Basic Model',
        'type': 'classification',
        'description': 'Quick 4-class tumor classification',
        'classes': {0: 'No Tumor', 1: 'Glioma', 2: 'Meningioma', 3: 'Pituitary'},
        'input_size': (64, 64),
        'channels': 1
    },
    'brain_detector': {
        'path': os.path.join(CODE_DIR, 'models', 'brain_tumor_detector.h5'),
        'name': 'Brain Tumor Detector',
        'type': 'classification',
        'description': 'General purpose tumor detector',
        'classes': {},
        'input_size': (240, 240),
        'channels': 3
    }
}

# Model cache
MODEL_CACHE = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_nifti_file(filepath):
    """Load a NIfTI file and extract a representative 2D slice"""
    if not NIFTI_SUPPORT:
        raise ImportError("nibabel is required for NIfTI support. Install with: pip install nibabel")
    
    try:
        # Load NIfTI file
        nifti_img = nib.load(filepath)
        img_data = nifti_img.get_fdata()
        
        # Handle different dimensions
        if img_data.ndim == 4:
            # 4D volume (x, y, z, modality) - take first modality
            img_data = img_data[:, :, :, 0]
        
        if img_data.ndim == 3:
            # 3D volume - extract middle slice from z-axis
            mid_slice = img_data.shape[2] // 2
            slice_2d = img_data[:, :, mid_slice]
        elif img_data.ndim == 2:
            # Already 2D
            slice_2d = img_data
        else:
            raise ValueError(f"Unexpected NIfTI dimensions: {img_data.ndim}")
        
        # Normalize to 0-255 range
        slice_2d = slice_2d.astype(np.float32)
        if slice_2d.max() > slice_2d.min():
            slice_2d = ((slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min()) * 255.0)
        else:
            slice_2d = np.zeros_like(slice_2d)
        
        # Convert to PIL Image
        slice_2d = slice_2d.astype(np.uint8)
        pil_img = Image.fromarray(slice_2d, mode='L')
        
        return pil_img
    
    except Exception as e:
        raise ValueError(f"Error loading NIfTI file: {str(e)}")

def get_available_models():
    """Get list of models that actually exist on disk"""
    available = {}
    for key, info in ALL_MODELS.items():
        if os.path.exists(info['path']):
            available[key] = {
                'name': info['name'],
                'type': info['type'],
                'description': info['description'],
                'classes': info['classes'],
                'input_size': info['input_size'],
                'channels': info['channels'],
                'size_mb': round(os.path.getsize(info['path']) / (1024*1024), 2)
            }
    return available

def load_model_cached(model_key):
    """Load and cache a model"""
    if model_key not in ALL_MODELS:
        raise ValueError(f"Unknown model key: {model_key}")
    
    if model_key in MODEL_CACHE:
        return MODEL_CACHE[model_key], ALL_MODELS[model_key]
    
    model_info = ALL_MODELS[model_key]
    model_path = model_info['path']
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading {model_info['name']}...")
    model = load_model(model_path)
    MODEL_CACHE[model_key] = model
    print(f"✓ Loaded {model_info['name']}")
    
    return model, model_info

def preprocess_for_model(img_bytes, model_info, filepath=None):
    """Preprocess image according to model requirements"""
    # Check if it's a NIfTI file
    if filepath and filepath.lower().endswith(('.nii', '.nii.gz')):
        if not NIFTI_SUPPORT:
            raise ImportError("nibabel is required for NIfTI support. Install with: pip install nibabel")
        img = load_nifti_file(filepath)
    else:
        # Load regular image
        img = Image.open(io.BytesIO(img_bytes))
    
    # Log original image info
    print(f"Original image: size={img.size}, mode={img.mode}")
    
    # Convert to required channels
    channels = model_info['channels']
    if channels == 1:
        img = img.convert('L')
    elif channels == 2:
        if img.mode in ('RGBA', 'LA'):
            img = img.convert('LA')
        else:
            gray = img.convert('L')
            img_la = Image.new('LA', gray.size)
            img_la.paste(gray, (0, 0))
            alpha_band = Image.new('L', gray.size, 255)
            img_la.putalpha(alpha_band)
            img = img_la
    else:
        img = img.convert('RGB')
    
    # Resize
    img = img.resize(model_info['input_size'], Image.Resampling.LANCZOS)
    arr = np.array(img).astype('float32')
    
    # Log array statistics before normalization
    print(f"Array before norm: shape={arr.shape}, min={arr.min():.2f}, max={arr.max():.2f}, mean={arr.mean():.2f}")
    
    # Normalize
    if channels == 2 and arr.ndim == 3:
        # Already 2 channels
        arr = arr / 255.0
    elif channels == 1:
        if arr.ndim == 2:
            arr = arr / 255.0
            arr = np.expand_dims(arr, axis=-1)
        else:
            arr = arr / 255.0
    else:
        arr = arr / 255.0
    
    # Batch dimension
    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=0)
    
    # Log final array statistics
    print(f"Array after norm: shape={arr.shape}, min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")
    
    return arr

def extract_segmentation_features(mask, probability_map):
    """Extract features from segmentation mask"""
    if mask is None or mask.sum() < 50:
        return {
            'tumor_present': False,
            'tumor_pixels': 0,
            'coverage_pct': 0.0
        }
    
    try:
        from skimage import measure
        
        labeled = measure.label(mask)
        regions = measure.regionprops(labeled)
        
        if not regions:
            return {
                'tumor_present': False,
                'tumor_pixels': 0,
                'coverage_pct': 0.0
            }
        
        largest = max(regions, key=lambda r: r.area)
        
        features = {
            'tumor_present': True,
            'tumor_pixels': int(mask.sum()),
            'coverage_pct': float(mask.sum() / mask.size * 100),
            'num_regions': len(regions),
            'largest_area': int(largest.area),
            'eccentricity': float(largest.eccentricity),
            'solidity': float(largest.solidity),
            'centroid': [float(largest.centroid[1]), float(largest.centroid[0])],
            'bbox': [int(v) for v in largest.bbox],
            'mean_confidence': float(np.mean(probability_map[mask == 1])) if probability_map is not None else 0.5,
            'max_confidence': float(np.max(probability_map)) if probability_map is not None else 0.5
        }
        
        return features
    except ImportError:
        return {
            'tumor_present': True,
            'tumor_pixels': int(mask.sum()),
            'coverage_pct': float(mask.sum() / mask.size * 100)
        }

def predict_survival_basic(features, patient_age):
    """Basic survival prediction based on heuristics"""
    if not features.get('tumor_present'):
        return {
            'prediction': 'N/A - No tumor detected',
            'confidence': 0.0,
            'method': 'heuristic'
        }
    
    # Heuristic based on tumor size and age
    coverage = features.get('coverage_pct', 0)
    age = patient_age if patient_age else 50  # Default
    
    # Risk scoring
    risk_score = 0
    if coverage > 15:
        risk_score += 3
    elif coverage > 8:
        risk_score += 2
    elif coverage > 3:
        risk_score += 1
    
    if age > 60:
        risk_score += 2
    elif age > 45:
        risk_score += 1
    
    # Survival categories
    if risk_score >= 4:
        survival = 'Short-term (<10 months)'
        confidence = 0.65
    elif risk_score >= 2:
        survival = 'Mid-term (10-15 months)'
        confidence = 0.60
    else:
        survival = 'Long-term (>15 months)'
        confidence = 0.70
    
    return {
        'prediction': survival,
        'confidence': confidence,
        'risk_score': risk_score,
        'method': 'heuristic'
    }

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'advanced_pipeline': ADVANCED_PIPELINE
    })

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get all available models"""
    available = get_available_models()
    return jsonify({
        'models': available,
        'total': len(available),
        'advanced_pipeline': ADVANCED_PIPELINE
    })

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Comprehensive analysis endpoint"""
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        
        # Get parameters
        model_key = request.form.get('model', 'best_segmentation')
        patient_age = request.form.get('age', None)
        if patient_age:
            try:
                patient_age = float(patient_age)
            except:
                patient_age = None
        
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        file.save(filepath)
        
        img_bytes = open(filepath, 'rb').read()
        
        # Load model
        model, model_info = load_model_cached(model_key)
        
        # Preprocess (pass filepath for NIfTI support)
        img_array = preprocess_for_model(img_bytes, model_info, filepath)
        
        # Run prediction
        print(f"Running prediction with {model_info['name']}...")
        predictions = model.predict(img_array, verbose=0)
        print(f"Prediction output shape: {predictions.shape}")
        print(f"Prediction stats: min={predictions.min():.4f}, max={predictions.max():.4f}, mean={predictions.mean():.4f}")
        
        # Build response
        response = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'filename': filename,
            'model_used': model_info['name'],
            'model_type': model_info['type'],
            'patient_age': patient_age
        }
        
        # Process based on model type
        if model_info['type'] == 'classification':
            # Classification output
            preds = np.asarray(predictions).ravel()
            top_k = min(10, preds.size)
            top_idx = preds.argsort()[::-1][:top_k]
            
            class_labels = model_info.get('classes', {})
            
            results = []
            for i in top_idx:
                class_name = class_labels.get(int(i), f'Class {int(i)}')
                results.append({
                    'class': int(i),
                    'class_name': class_name,
                    'probability': float(preds[i])
                })
            
            predicted_idx = int(np.argmax(preds))
            predicted_name = class_labels.get(predicted_idx, f'Class {predicted_idx}')
            
            response['classification'] = {
                'tumor_detected': predicted_name != 'No Tumor' and predicted_name != 'Healthy/Normal',
                'predicted_class': predicted_name,
                'confidence': float(np.max(preds)),
                'all_predictions': results[:5]
            }
            
            response['segmentation'] = None
            response['features'] = None
            response['survival'] = None
            
        else:  # segmentation
            # Segmentation output
            raw_pred = predictions[0]
            if raw_pred.ndim == 3:
                raw_pred = raw_pred[:, :, 0]
            
            probability_map = np.clip(raw_pred, 0, 1)
            
            # Use adaptive thresholding based on prediction statistics
            pred_mean = probability_map.mean()
            pred_std = probability_map.std()
            
            # If predictions are very uniform (low std), likely no tumor
            if pred_std < 0.1:
                # Very uniform predictions - likely no distinct tumor
                threshold = 0.95  # Very High threshold
            elif pred_mean > 0.7:
                # High mean - model is uncertain
                threshold = 0.8
            else:
                # Normal case - use stricter threshold
                threshold = 0.65
            
            mask = (probability_map >= threshold).astype(np.uint8)
            
            print(f"Prediction stats: mean={pred_mean:.4f}, std={pred_std:.4f}, threshold={threshold:.2f}")
            print(f"Segmentation: mask pixels={mask.sum()}, coverage={(mask.sum()/mask.size)*100:.2f}%")
            
            # Extract features
            features = extract_segmentation_features(mask, probability_map)
            
            print(f"Features extracted: tumor_present={features.get('tumor_present', False)}, coverage={features.get('coverage_pct', 0):.2f}%")
            
            # Classification based on segmentation
            # Require at least 2% coverage to be confident it's a tumor (filters noise)
            coverage = features.get('coverage_pct', 0)
            tumor_detected = features.get('tumor_present', False) and coverage > 3.0
            
            # Infer tumor type from features
            if tumor_detected:
                if coverage > 15:
                    tumor_type = 'Glioma (likely high-grade)'
                elif coverage > 8:
                    tumor_type = 'Glioma (moderate grade)'
                elif coverage < 5:
                    tumor_type = 'Pituitary adenoma (small, localized)'
                else:
                    tumor_type = 'Meningioma or low-grade glioma'
            else:
                tumor_type = 'No Tumor'
            
            response['classification'] = {
                'tumor_detected': tumor_detected,
                'predicted_class': tumor_type,
                'confidence': features.get('mean_confidence', 0.5),
                'inference_method': 'segmentation_based'
            }
            
            response['segmentation'] = {
                'success': True,
                'coverage': coverage,
                'tumor_pixels': features.get('tumor_pixels', 0)
            }
            
            response['features'] = features
            
            # Survival prediction
            if patient_age and tumor_detected:
                if ADVANCED_PIPELINE:
                    try:
                        survival_result = predict_survival_from_features(features, patient_age)
                        response['survival'] = survival_result
                    except:
                        response['survival'] = predict_survival_basic(features, patient_age)
                else:
                    response['survival'] = predict_survival_basic(features, patient_age)
            else:
                response['survival'] = None
        
        # Summary
        response['summary'] = {
            'tumor_present': response['classification']['tumor_detected'],
            'primary_diagnosis': response['classification']['predicted_class'],
            'confidence': response['classification']['confidence'],
            'recommendations': []
        }
        
        if response['classification']['tumor_detected']:
            response['summary']['recommendations'].append('Tumor detected - Recommend further medical evaluation')
            response['summary']['recommendations'].append('Consult with oncologist for treatment plan')
            if response.get('survival'):
                surv = response['survival'].get('prediction', '')
                response['summary']['recommendations'].append(f'Estimated survival: {surv}')
        else:
            response['summary']['recommendations'].append('No significant tumor detected')
            response['summary']['recommendations'].append('Continue regular screening as per medical advice')
        
        # Cleanup
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        print(f"Error: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e),
            'trace': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    print("=" * 80)
    print("🧠 Enhanced Brain Tumor Detection System")
    print("=" * 80)
    print(f"Available models: {len(get_available_models())}")
    print(f"Advanced pipeline: {ADVANCED_PIPELINE}")
    print("=" * 80)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
