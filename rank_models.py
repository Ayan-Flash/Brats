import os
import sys
import numpy as np
from PIL import Image
import tensorflow as tf
import io

# Add current directory to path
sys.path.append(os.getcwd())

# Import from app
from app import ALL_MODELS, preprocess_for_model, load_model_cached

def rank_models():
    print("=" * 60)
    print("RANKING MODELS")
    print("=" * 60)
    
    tumor_img_path = 'test_samples/real_tumor.png'
    no_tumor_img_path = 'test_samples/real_no_tumor.png'
    
    if not os.path.exists(tumor_img_path) or not os.path.exists(no_tumor_img_path):
        print("Test images not found!")
        return

    # Load images bytes
    with open(tumor_img_path, 'rb') as f:
        tumor_bytes = f.read()
    with open(no_tumor_img_path, 'rb') as f:
        no_tumor_bytes = f.read()
        
    results = []
    
    for key, info in ALL_MODELS.items():
        print(f"\nTesting {info['name']} ({key})...")
        
        if not os.path.exists(info['path']):
            print(f"  Model file not found: {info['path']}")
            continue
            
        try:
            # Load model
            model = tf.keras.models.load_model(info['path'], compile=False)
            
            # Test Tumor Image
            input_tumor = preprocess_for_model(tumor_bytes, info)
            pred_tumor = model.predict(input_tumor, verbose=0)
            
            # Test No-Tumor Image
            input_no_tumor = preprocess_for_model(no_tumor_bytes, info)
            pred_no_tumor = model.predict(input_no_tumor, verbose=0)
            
            score = 0
            details = {}
            
            if info['type'] == 'segmentation':
                # Tumor Image: Expect high coverage
                mask_tumor = (pred_tumor[0, :, :, 0] > 0.5).astype(int)
                tumor_coverage = np.mean(mask_tumor) * 100
                
                # No-Tumor Image: Expect low coverage
                mask_no_tumor = (pred_no_tumor[0, :, :, 0] > 0.5).astype(int)
                no_tumor_coverage = np.mean(mask_no_tumor) * 100
                
                # Scoring: Higher tumor coverage + Lower false positive coverage
                # Ideal: Tumor > 10%, No-Tumor < 1%
                score = tumor_coverage - (no_tumor_coverage * 5) # Penalize false positives heavily
                
                details = {
                    'tumor_coverage': f"{tumor_coverage:.2f}%",
                    'false_positive': f"{no_tumor_coverage:.2f}%"
                }
                
            elif info['type'] == 'classification':
                # Tumor Image: Expect 'Tumor' class (index 1 usually, or specific class)
                # No-Tumor Image: Expect 'No Tumor' class (index 0)
                
                # Simplified scoring for classification
                p_tumor = np.max(pred_tumor)
                p_no_tumor = np.max(pred_no_tumor)
                
                # This is harder to score generically without knowing exact class mapping
                # Assuming index 0 is No Tumor
                
                score = 0 # Placeholder
                details = {'note': 'Classification ranking skipped for now'}
            
            results.append({
                'key': key,
                'name': info['name'],
                'score': score,
                'details': details
            })
            print(f"  Score: {score:.2f} {details}")
            
        except Exception as e:
            print(f"  Failed: {e}")
            
    # Sort and Rank
    print("\n" + "=" * 60)
    print("FINAL RANKING")
    print("=" * 60)
    
    results.sort(key=lambda x: x['score'], reverse=True)
    
    for i, res in enumerate(results):
        print(f"{i+1}. {res['name']} (Score: {res['score']:.2f})")
        
    print("\nTop 3 Models:")
    top3 = [r['key'] for r in results[:3]]
    print(top3)
    
    return top3

if __name__ == "__main__":
    rank_models()
