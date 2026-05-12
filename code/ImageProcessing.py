import openslide
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.color import rgb2gray
import torch
from torchvision import models, transforms
import os
import glob
from tqdm import tqdm
import warnings
from concurrent.futures import ThreadPoolExecutor
import gc
warnings.filterwarnings('ignore')

# =============================================
# CONFIGURATION
# =============================================
class Config:
    # Paths
    SVS_DIR = "images/"
    VALID_PATIENTS_FILE = "csv/common_82_samples.txt"
    OUTPUT_TRADITIONAL = "image_features_traditional.csv"
    OUTPUT_DL = "image_features_deeplearning.csv"
    
    # Patch extraction
    PATCH_SIZE = 256
    PATCHES_PER_SLIDE = 500  # Increased for better coverage
    TISSUE_THRESHOLD = 0.3   # Minimum tissue percentage
    DETECTION_LEVEL = 5      # Low-res level for tissue detection
    
    # Deep learning
    BATCH_SIZE = 32          # Process DL features in batches
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================
# SLIDE LOADING & VALIDATION
# =============================================
def load_valid_patients(valid_patients_file):
    """Load list of valid patient IDs"""
    with open(valid_patients_file, 'r') as f:
        valid_patients = set(line.strip() for line in f if line.strip())
    print(f"✓ Loaded {len(valid_patients)} valid patient IDs")
    return valid_patients

def get_valid_slides(svs_dir, valid_patients):
    """Get SVS files that match valid patients"""
    all_svs = glob.glob(os.path.join(svs_dir, "*.svs"))
    valid_slides = []
    skipped = []
    
    for svs_path in all_svs:
        slide_id = os.path.basename(svs_path).replace('.svs', '')
        patient_id = slide_id[:12]
        
        if patient_id in valid_patients:
            valid_slides.append(svs_path)
        else:
            skipped.append(slide_id)
    
    print(f"✓ Found {len(valid_slides)} valid slides out of {len(all_svs)} total")
    print(f"  Skipped {len(skipped)} slides (patients without complete omics data)")
    return valid_slides

# =============================================
# TISSUE DETECTION (OPTIMIZED)
# =============================================
def detect_tissue_mask(slide, level=None):
    """
    Create tissue mask at appropriate level
    Returns: mask, scale factors
    """
    if level is None:
        level = slide.level_count - 4  # Lower resolution for speed
    
    level = max(0, min(level, slide.level_count - 1))
    
    # Read at detection level
    dims = slide.level_dimensions[level]
    img = slide.read_region((0, 0), level, dims)
    img_rgb = np.array(img.convert('RGB'))
    
    # Convert to HSV for better tissue detection
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    
    # Otsu thresholding on saturation channel
    _, mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Calculate scale factors
    full_w, full_h = slide.dimensions
    mask_h, mask_w = mask.shape
    scale_x = full_w / mask_w
    scale_y = full_h / mask_h
    
    return mask, scale_x, scale_y

# =============================================
# SMART PATCH EXTRACTION
# =============================================
def extract_patches_smart(slide, patch_size=256, n_patches=500, tissue_threshold=0.3):
    """
    Extract patches with intelligent sampling for maximum tissue coverage
    """
    # Get tissue mask
    mask, scale_x, scale_y = detect_tissue_mask(slide)
    
    patches = []
    coordinates = []
    tissues_regions = []
    
    full_w, full_h = slide.dimensions
    
    # Find tissue contours for targeted extraction
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate total tissue area for weighting
    total_tissue_area = sum(cv2.contourArea(c) for c in contours)
    
    if total_tissue_area == 0:
        return [], []
    
    # Distribute patches proportionally to tissue region sizes
    for contour in contours:
        area = cv2.contourArea(contour)
        n_patches_region = max(1, int(n_patches * area / total_tissue_area))
        tissues_regions.append((contour, area, n_patches_region))
    
    # Extract patches from each tissue region
    for contour, area, n_region in tqdm(tissues_regions, desc="  Tissue regions", leave=False):
        # Get bounding box of contour in mask coordinates
        x, y, w, h = cv2.boundingRect(contour)
        
        # Convert to full-resolution coordinates
        x_full = int(x * scale_x)
        y_full = int(y * scale_y)
        w_full = int(w * scale_x)
        h_full = int(h * scale_y)
        
        # Generate candidate positions within this region
        candidates_x = list(range(x_full, min(x_full + w_full, full_w - patch_size), patch_size // 2))
        candidates_y = list(range(y_full, min(y_full + h_full, full_h - patch_size), patch_size // 2))
        
        if not candidates_x or not candidates_y:
            continue
        
        # Sample positions
        np.random.seed(42)
        n_candidates = len(candidates_x) * len(candidates_y)
        n_sample = min(n_region, n_candidates)
        
        if n_candidates > 0:
            indices = np.random.choice(n_candidates, n_sample, replace=False)
            
            for idx in indices:
                ix = idx % len(candidates_x)
                iy = idx // len(candidates_x)
                
                px = candidates_x[ix]
                py = candidates_y[iy]
                
                try:
                    # Read patch at full resolution
                    patch = slide.read_region((px, py), 0, (patch_size, patch_size))
                    patch_rgb = np.array(patch.convert('RGB'))
                    
                    patches.append(patch_rgb)
                    coordinates.append((px, py))
                except:
                    continue
    
    return patches, coordinates

# =============================================
# TRADITIONAL FEATURE EXTRACTION
# =============================================
def extract_traditional_features(patch):
    """
    Extract comprehensive traditional computer vision features
    """
    try:
        gray = rgb2gray(patch)
        gray_uint = (gray * 255).astype(np.uint8)
        
        features = []
        
        # 1. Color features (RGB statistics)
        for channel in range(3):
            ch = patch[:, :, channel]
            features.extend([
                np.mean(ch), np.std(ch), np.percentile(ch, 25),
                np.percentile(ch, 75), np.percentile(ch, 90)
            ])
        
        # 2. Texture - LBP
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
        features.extend(lbp_hist)
        
        # 3. Texture - GLCM (multiple angles)
        distances = [1, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        for d in distances:
            glcm = graycomatrix(gray_uint, distances=[d], angles=angles, levels=256, symmetric=True, normed=True)
            features.extend([
                graycoprops(glcm, 'contrast').mean(),
                graycoprops(glcm, 'dissimilarity').mean(),
                graycoprops(glcm, 'homogeneity').mean(),
                graycoprops(glcm, 'energy').mean(),
                graycoprops(glcm, 'correlation').mean()
            ])
        
        # 4. Edge features
        edges = cv2.Canny(gray_uint, 50, 150)
        features.append(np.mean(edges) / 255)  # Edge density
        
        # 5. Nuclei-like features (color deconvolution approximation)
        # Hematoxylin channel (blue/purple)
        hematoxylin = 1.0 * patch[:, :, 2] - 0.5 * patch[:, :, 0] - 0.5 * patch[:, :, 1]
        features.extend([np.mean(hematoxylin), np.std(hematoxylin)])
        
        # Eosin channel (pink)
        eosin = 1.0 * patch[:, :, 0] + 0.5 * patch[:, :, 1] - 1.0 * patch[:, :, 2]
        features.extend([np.mean(eosin), np.std(eosin)])
        
        return np.array(features, dtype=np.float32)
    
    except Exception as e:
        return np.zeros(38, dtype=np.float32)  # Return zeros if extraction fails

# =============================================
# DEEP LEARNING FEATURE EXTRACTION
# =============================================
class DLFeatureExtractor:
    """Efficient batch-based deep learning feature extractor"""
    
    def __init__(self, model_name='resnet50', device='cpu'):
        self.device = device
        
        # Load pre-trained model
        if model_name == 'resnet50':
            model = models.resnet50(weights='IMAGENET1K_V1')
            self.model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove FC layer
            self.feature_dim = 2048
        elif model_name == 'densenet121':
            model = models.densenet121(weights='IMAGENET1K_V1')
            self.model = model.features
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.feature_dim = 1024
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # Preprocessing pipeline
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✓ DL Extractor initialized on {device} with {model_name}")
        print(f"  Feature dimension: {self.feature_dim}")
    
    def extract_batch(self, patches):
        """Extract features from a batch of patches"""
        batch_tensors = []
        
        for patch in patches:
            patch_pil = Image.fromarray(patch)
            tensor = self.preprocess(patch_pil)
            batch_tensors.append(tensor)
        
        batch = torch.stack(batch_tensors).to(self.device)
        
        with torch.no_grad():
            features = self.model(batch)
            features = features.squeeze(-1).squeeze(-1)  # Remove spatial dimensions
            return features.cpu().numpy()

# =============================================
# FEATURE DATAFRAME CREATION
# =============================================
def create_feature_dataframe(patches, slide_id, coordinates, feature_type='traditional', dl_extractor=None):
    """Create DataFrame with features"""
    features_list = []
    
    if feature_type == 'traditional':
        # Process traditional features
        for idx, patch in enumerate(patches):
            features = extract_traditional_features(patch)
            features_dict = {
                'patient_id': slide_id[:12],
                'slide_id': slide_id,
                'tile_x': coordinates[idx][0],
                'tile_y': coordinates[idx][1],
                'tile_idx': idx
            }
            # Add features
            for f_idx, feat in enumerate(features):
                features_dict[f'feat_{f_idx:04d}'] = feat
            features_list.append(features_dict)
    
    elif feature_type == 'dl' and dl_extractor is not None:
        # Process deep learning features in batches
        batch_size = 32
        for start_idx in range(0, len(patches), batch_size):
            end_idx = min(start_idx + batch_size, len(patches))
            batch_patches = patches[start_idx:end_idx]
            batch_features = dl_extractor.extract_batch(batch_patches)
            
            for i, features in enumerate(batch_features):
                idx = start_idx + i
                features_dict = {
                    'patient_id': slide_id[:12],
                    'slide_id': slide_id,
                    'tile_x': coordinates[idx][0],
                    'tile_y': coordinates[idx][1],
                    'tile_idx': idx
                }
                # Add DL features
                for f_idx, feat in enumerate(features):
                    features_dict[f'dl_feat_{f_idx:04d}'] = feat
                features_list.append(features_dict)
    
    return pd.DataFrame(features_list)

# =============================================
# MAIN PROCESSING PIPELINE
# =============================================
def process_slides_for_features(svs_paths, output_file, feature_type='traditional', dl_extractor=None):
    """
    Process multiple slides and extract features
    
    Args:
        svs_paths: List of SVS file paths
        output_file: Output CSV path
        feature_type: 'traditional' or 'dl'
        dl_extractor: DLFeatureExtractor instance (for DL features)
    """
    print(f"\n{'='*60}")
    print(f"EXTRACTING {feature_type.upper()} FEATURES")
    print(f"{'='*60}")
    
    all_dataframes = []
    processing_stats = []
    
    for svs_path in tqdm(svs_paths, desc=f"Processing slides ({feature_type})"):
        slide_id = os.path.basename(svs_path).replace('.svs', '')
        patient_id = slide_id[:12]
        
        try:
            # Open slide
            slide = openslide.OpenSlide(svs_path)
            
            # Extract patches
            patches, coords = extract_patches_smart(
                slide, 
                patch_size=Config.PATCH_SIZE,
                n_patches=Config.PATCHES_PER_SLIDE,
                tissue_threshold=Config.TISSUE_THRESHOLD
            )
            
            if len(patches) > 0:
                # Extract features and create DataFrame
                features_df = create_feature_dataframe(
                    patches, slide_id, coords, 
                    feature_type=feature_type, 
                    dl_extractor=dl_extractor
                )
                all_dataframes.append(features_df)
                
                processing_stats.append({
                    'slide_id': slide_id,
                    'patient_id': patient_id,
                    'num_patches': len(patches),
                    'features_extracted': len(features_df),
                    'status': 'SUCCESS'
                })
            else:
                processing_stats.append({
                    'slide_id': slide_id,
                    'patient_id': patient_id,
                    'num_patches': 0,
                    'features_extracted': 0,
                    'status': 'NO_TISSUE'
                })
            
            slide.close()
            gc.collect()  # Free memory
            
        except Exception as e:
            print(f"\n  ✗ Error: {slide_id} - {str(e)}")
            processing_stats.append({
                'slide_id': slide_id,
                'patient_id': patient_id,
                'status': f'ERROR: {str(e)}'
            })
    
    # Combine all features
    if all_dataframes:
        final_df = pd.concat(all_dataframes, ignore_index=True)
        final_df.to_csv(output_file, index=False)
        
        # Print summary
        n_slides_success = sum(1 for s in processing_stats if s['status'] == 'SUCCESS')
        total_patches = sum(s.get('num_patches', 0) for s in processing_stats)
        
        print(f"\n{'='*60}")
        print(f"{feature_type.upper()} FEATURES EXTRACTION COMPLETE")
        print(f"{'='*60}")
        print(f"✓ Successful slides: {n_slides_success}/{len(svs_paths)}")
        print(f"✓ Total patches extracted: {total_patches}")
        print(f"✓ Total feature vectors: {len(final_df)}")
        print(f"✓ Feature dimension: {len([c for c in final_df.columns if 'feat_' in c])}")
        print(f"✓ Output saved: {output_file}")
        
        # Save processing log
        log_df = pd.DataFrame(processing_stats)
        log_df.to_csv(output_file.replace('.csv', '_log.csv'), index=False)
        
        return final_df
    else:
        print(f"\n✗ No features extracted!")
        return pd.DataFrame()

# =============================================
# MAIN EXECUTION
# =============================================
if __name__ == "__main__":
    print("TCGA Integrative Analysis - Image Feature Extraction")
    print("="*60)
    
    # Load valid patients
    valid_patients = load_valid_patients(Config.VALID_PATIENTS_FILE)
    
    # Get valid slides
    valid_slides = get_valid_slides(Config.SVS_DIR, valid_patients)
    
    print(f"\nConfiguration:")
    print(f"  Patch size: {Config.PATCH_SIZE}x{Config.PATCH_SIZE}")
    print(f"  Patches per slide: {Config.PATCHES_PER_SLIDE}")
    print(f"  Tissue threshold: {Config.TISSUE_THRESHOLD}")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Estimated total patches: {len(valid_slides) * Config.PATCHES_PER_SLIDE:,}")
    
    # Phase 1: Traditional Features
    print(f"\n{'#'*60}")
    print(f"PHASE 1: TRADITIONAL COMPUTER VISION FEATURES")
    print(f"{'#'*60}")
    
    traditional_features = process_slides_for_features(
        svs_paths=valid_slides,
        output_file=Config.OUTPUT_TRADITIONAL,
        feature_type='traditional'
    )
    
    # Phase 2: Deep Learning Features
    print(f"\n{'#'*60}")
    print(f"PHASE 2: DEEP LEARNING FEATURES (Transfer Learning)")
    print(f"{'#'*60}")
    
    # Initialize DL extractor
    dl_extractor = DLFeatureExtractor(
        model_name='resnet50', 
        device=Config.DEVICE
    )
    
    dl_features = process_slides_for_features(
        svs_paths=valid_slides,
        output_file=Config.OUTPUT_DL,
        feature_type='dl',
        dl_extractor=dl_extractor
    )
    
    # Final summary
    print(f"\n{'#'*60}")
    print(f"COMPLETE PIPELINE SUMMARY")
    print(f"{'#'*60}")
    
    if len(traditional_features) > 0:
        trad_dim = len([c for c in traditional_features.columns if 'feat_' in c])
        print(f"Traditional Features:")
        print(f"  Shape: {traditional_features.shape}")
        print(f"  Feature dimension: {trad_dim}")
        print(f"  Saved to: {Config.OUTPUT_TRADITIONAL}")
    
    if len(dl_features) > 0:
        dl_dim = len([c for c in dl_features.columns if 'dl_feat_' in c])
        print(f"\nDeep Learning Features:")
        print(f"  Shape: {dl_features.shape}")
        print(f"  Feature dimension: {dl_dim}")
        print(f"  Saved to: {Config.OUTPUT_DL}")
    
    print(f"\n✓ All processing complete!")