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
import json
from tqdm import tqdm
import warnings
import gc
import pickle
from pathlib import Path
import time
warnings.filterwarnings('ignore')

# =============================================
# CONFIGURATION
# =============================================
class Config:
    # Paths
    SVS_DIR = "images/"
    VALID_PATIENTS_FILE = "csv/common_82_samples.txt"
    CHECKPOINT_DIR = "checkpoints/"
    OUTPUT_DIR = "output_features/"
    
    # Patch extraction parameters
    PATCH_SIZE = 256
    PATCHES_PER_SLIDE = 1000
    TISSUE_THRESHOLD = 0.3
    
    # Memory management (CRITICAL!)
    SAVE_INTERVAL = 1          # Save checkpoints every N slides
    MAX_PATCHES_IN_MEMORY = 100 # Process patches in chunks to avoid OOM
    BATCH_SIZE_DL = 16         # Smaller batch for DL (CPU memory)
    
    # Processing levels
    EXTRACTION_LEVEL = 2       # Extract patches from level 2 (smaller but still detailed)
    DETECTION_LEVEL = 5        # Tissue detection at very low res
    
    # Feature type to run
    RUN_TRADITIONAL = True
    RUN_DL = True             # Set to True later after traditional works
    DEVICE = "cpu"
    
    # Output files
    OUTPUT_TRADITIONAL = "output_features/image_features_traditional.csv"
    OUTPUT_DL = "output_features/image_features_deeplearning.csv"

# =============================================
# CHECKPOINT MANAGEMENT
# =============================================
class CheckpointManager:
    """Manages checkpointing for resumable processing"""
    
    def __init__(self, checkpoint_dir, feature_type):
        self.checkpoint_dir = Path(checkpoint_dir) / feature_type
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.checkpoint_dir / "processing_state.json"
        self.data_dir = self.checkpoint_dir / "partial_data"
        self.data_dir.mkdir(exist_ok=True)
        
        # Load existing state if available
        self.state = self.load_state()
    
    def load_state(self):
        """Load processing state from checkpoint"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            'processed_slides': [],
            'failed_slides': {},
            'total_patches': 0,
            'last_slide_idx': -1
        }
    
    def save_state(self):
        """Save processing state to checkpoint"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def mark_slide_complete(self, slide_id, num_patches):
        """Mark a slide as successfully processed"""
        if slide_id not in self.state['processed_slides']:
            self.state['processed_slides'].append(slide_id)
        self.state['total_patches'] += num_patches
        self.save_state()
    
    def mark_slide_failed(self, slide_id, error):
        """Mark a slide as failed"""
        self.state['failed_slides'][slide_id] = str(error)
        self.save_state()
    
    def save_partial_features(self, slide_id, features_df):
        """Save features for a single slide"""
        filepath = self.data_dir / f"{slide_id}.parquet"
        features_df.to_parquet(filepath, index=False)
    
    def load_all_partial_features(self):
        """Load all previously saved partial features"""
        all_files = list(self.data_dir.glob("*.parquet"))
        if not all_files:
            return pd.DataFrame()
        
        dfs = []
        for file in tqdm(all_files, desc="Loading checkpoint data"):
            dfs.append(pd.read_parquet(file))
        
        return pd.concat(dfs, ignore_index=True)
    
    def is_slide_processed(self, slide_id):
        """Check if slide was already processed"""
        return slide_id in self.state['processed_slides']

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
    
    for svs_path in sorted(all_svs):  # Sort for consistent ordering
        slide_id = os.path.basename(svs_path).replace('.svs', '')
        patient_id = slide_id[:12]
        
        if patient_id in valid_patients:
            valid_slides.append(svs_path)
        else:
            skipped.append(slide_id)
    
    print(f"✓ Found {len(valid_slides)} valid slides out of {len(all_svs)} total")
    return valid_slides, skipped

# =============================================
# MEMORY-EFFICIENT TISSUE DETECTION
# =============================================
def detect_tissue_mask(slide, detection_level=5):
    """Create tissue mask at low resolution"""
    level = min(detection_level, slide.level_count - 1)
    
    dims = slide.level_dimensions[level]
    # Read only at small size
    img = slide.read_region((0, 0), level, dims)
    
    # Resize to even smaller for speed
    small_size = (min(dims[0], 1000), min(dims[1], 1000))
    img_small = img.resize(small_size, Image.Resampling.LANCZOS)
    img_rgb = np.array(img_small.convert('RGB'))
    
    # Use HSV for better tissue detection
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    
    # Threshold
    _, mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Quick morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Calculate scale factors
    full_w, full_h = slide.dimensions
    scale_x = full_w / mask.shape[1]
    scale_y = full_h / mask.shape[0]
    
    return mask, scale_x, scale_y

# =============================================
# MEMORY-EFFICIENT PATCH EXTRACTION
# =============================================
def extract_patches_generator(slide, patch_size=256, n_patches=500, 
                             extraction_level=2, tissue_threshold=0.3):
    """
    Extract patches with YIELD to save memory.
    Uses lower resolution level for patches.
    """
    # Get tissue mask at low resolution
    mask, scale_x, scale_y = detect_tissue_mask(slide)
    
    # Find tissue contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return
    
    # Calculate total tissue area
    total_area = sum(cv2.contourArea(c) for c in contours)
    if total_area == 0:
        return
    
    # Distribute patches proportionally
    patches_yielded = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        n_region = max(1, int(n_patches * area / total_area))
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Convert to extraction level coordinates
        downsample = slide.level_downsamples[extraction_level]
        x_full = int(x * scale_x / downsample)
        y_full = int(y * scale_y / downsample)
        w_full = int(w * scale_x / downsample)
        h_full = int(h * scale_y / downsample)
        
        # Get dimensions at extraction level
        ext_dims = slide.level_dimensions[extraction_level]
        
        # Generate random positions within region
        np.random.seed(int(time.time() * 1000) % 10000)  # Different seed per run
        
        attempts = 0
        max_attempts = n_region * 3
        
        while patches_yielded < n_patches and attempts < max_attempts:
            px = np.random.randint(x_full, min(x_full + w_full - patch_size, ext_dims[0] - patch_size)) if x_full + w_full > patch_size else x_full
            py = np.random.randint(y_full, min(y_full + h_full - patch_size, ext_dims[1] - patch_size)) if y_full + h_full > patch_size else y_full
            
            if px < 0 or py < 0 or px + patch_size >= ext_dims[0] or py + patch_size >= ext_dims[1]:
                attempts += 1
                continue
            
            try:
                # Read at extraction level (not full res!)
                patch = slide.read_region(
                    (int(px * downsample), int(py * downsample)), 
                    0, 
                    (patch_size, patch_size)
                )
                patch_rgb = np.array(patch.convert('RGB'))
                
                # Quick tissue check
                if np.mean(patch_rgb) < 240 and np.std(patch_rgb) > 15:  # Not white background
                    patches_yielded += 1
                    yield patch_rgb, (px, py)
                
            except:
                pass
            
            attempts += 1

# =============================================
# TRADITIONAL FEATURE EXTRACTION (OPTIMIZED)
# =============================================
def extract_traditional_features(patch):
    """Fast traditional feature extraction"""
    try:
        # Convert to proper types
        gray_uint = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        gray_float = gray_uint.astype(np.float32) / 255.0
        
        features = []
        
        # 1. Intensity statistics (6 features)
        features.extend([np.mean(gray_float), np.std(gray_float), 
                        np.median(gray_float), np.percentile(gray_float, 25),
                        np.percentile(gray_float, 75), np.percentile(gray_float, 90)])
        
        # 2. RGB channel statistics (5 features per channel = 15 features)
        for i in range(3):
            ch = patch[:, :, i].astype(np.float32)
            features.extend([np.mean(ch), np.std(ch), np.median(ch)])
        
        # 3. Edge features (2 features)
        edges = cv2.Canny(gray_uint, 50, 150)
        features.extend([np.mean(edges) / 255.0, np.sum(edges > 0) / edges.size])
        
        # 4. Blob-like features (Laplacian) (2 features)
        laplacian = cv2.Laplacian(gray_uint, cv2.CV_64F)
        features.extend([np.mean(np.abs(laplacian)), np.std(laplacian)])
        
        # 5. Gradient features (Sobel) (4 features)
        sobelx = cv2.Sobel(gray_uint, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_uint, cv2.CV_64F, 0, 1, ksize=3)
        features.extend([np.mean(np.abs(sobelx)), np.std(sobelx),
                        np.mean(np.abs(sobely)), np.std(sobely)])
        
        # 6. Local binary pattern (10 features)
        lbp = local_binary_pattern(gray_float, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
        features.extend(lbp_hist)
        
        return np.array(features, dtype=np.float32)
    
    except Exception as e:
        return np.zeros(39, dtype=np.float32)

# =============================================
# DEEP LEARNING FEATURE EXTRACTOR (MEMORY EFFICIENT)
# =============================================
class LightweightDLExtractor:
    """Memory-efficient DL extractor"""
    
    def __init__(self, device='cpu'):
        self.device = device
        
        # Use ResNet18 instead of ResNet50 for CPU (faster, less memory)
        model = models.resnet18(weights='IMAGENET1K_V1')
        self.model = torch.nn.Sequential(*list(model.children())[:-1])
        self.feature_dim = 512
        self.model = self.model.to(device)
        self.model.eval()
        
        self.preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✓ Lightweight DL Extractor: ResNet18 on {device} (dim={self.feature_dim})")
    
    def extract_batch(self, patches):
        """Extract features from batch"""
        if not patches:
            return np.array([])
        
        batch_tensors = []
        for patch in patches:
            try:
                patch_pil = Image.fromarray(patch.astype(np.uint8))
                tensor = self.preprocess(patch_pil)
                batch_tensors.append(tensor)
            except:
                # Create blank tensor if patch is bad
                batch_tensors.append(torch.zeros(3, 224, 224))
        
        if not batch_tensors:
            return np.array([])
        
        batch = torch.stack(batch_tensors).to(self.device)
        
        with torch.no_grad():
            features = self.model(batch)
            features = features.squeeze(-1).squeeze(-1)
            return features.cpu().numpy()

# =============================================
# MAIN PROCESSING PIPELINE (WITH CHECKPOINTS)
# =============================================
def process_slide_memory_efficient(slide_path, slide_id, patch_size, n_patches, 
                                  extraction_level, checkpoint_mgr, feature_type='traditional',
                                  dl_extractor=None):
    """Process a single slide with memory efficiency"""
    
    # Process in CHUNKS to avoid memory overload
    patches_buffer = []
    coords_buffer = []
    all_features = []
    
    slide = None
    
    try:
        slide = openslide.OpenSlide(slide_path)
        print(f"\n[{slide_id}] Opened successfully (dims={slide.dimensions})")
        
        patch_gen = extract_patches_generator(
            slide, 
            patch_size=patch_size,
            n_patches=n_patches,
            extraction_level=extraction_level,
            tissue_threshold=Config.TISSUE_THRESHOLD
        )
        
        chunk_count = 0
        for patch, coord in patch_gen:
            patches_buffer.append(patch)
            coords_buffer.append(coord)
            
            # Process when buffer is full or we've got enough
            if len(patches_buffer) >= Config.MAX_PATCHES_IN_MEMORY:
                # Extract features for this chunk
                if feature_type == 'traditional':
                    chunk_features = [extract_traditional_features(p) for p in patches_buffer]
                else:
                    chunk_features = dl_extractor.extract_batch(patches_buffer)
                
                # Create mini dataframe
                for i, features in enumerate(chunk_features):
                    feature_dict = {
                        'patient_id': slide_id[:12],
                        'slide_id': slide_id,
                        'tile_x': coords_buffer[i][0],
                        'tile_y': coords_buffer[i][1],
                        'chunk': chunk_count
                    }
                    for j, feat in enumerate(features):
                        feature_dict[f'feat_{j:04d}'] = feat
                    all_features.append(feature_dict)
                
                chunk_count += 1
                print(f"  Chunk {chunk_count}: Processed {len(patches_buffer)} patches", end='\r')
                
                # Clear buffers
                patches_buffer.clear()
                coords_buffer.clear()
                gc.collect()
        
        # Process remaining patches
        if patches_buffer:
            if feature_type == 'traditional':
                chunk_features = [extract_traditional_features(p) for p in patches_buffer]
            else:
                chunk_features = dl_extractor.extract_batch(patches_buffer)
            
            for i, features in enumerate(chunk_features):
                feature_dict = {
                    'patient_id': slide_id[:12],
                    'slide_id': slide_id,
                    'tile_x': coords_buffer[i][0],
                    'tile_y': coords_buffer[i][1],
                    'chunk': chunk_count
                }
                for j, feat in enumerate(features):
                    feature_dict[f'feat_{j:04d}'] = feat
                all_features.append(feature_dict)
        
        # Create final DataFrame for this slide
        if all_features:
            slide_df = pd.DataFrame(all_features)
            print(f"\n[{slide_id}] ✓ Extracted {len(slide_df)} total feature vectors")
            
            # Save to checkpoint
            checkpoint_mgr.save_partial_features(slide_id, slide_df)
            checkpoint_mgr.mark_slide_complete(slide_id, len(slide_df))
            
            del slide_df, all_features
            gc.collect()
            
            return True, len(all_features)
        else:
            print(f"\n[{slide_id}] ⚠ No features extracted")
            return False, 0
    
    except Exception as e:
        print(f"\n[{slide_id}] ✗ Error: {str(e)}")
        checkpoint_mgr.mark_slide_failed(slide_id, str(e))
        return False, 0
    
    finally:
        if slide:
            slide.close()

def run_processing_pipeline(feature_type='traditional'):
    """Main processing pipeline with checkpoint support"""
    
    print(f"\n{'#'*60}")
    print(f"PROCESSING: {feature_type.upper()} FEATURES")
    print(f"{'#'*60}")
    
    # Setup checkpoint manager
    checkpoint_mgr = CheckpointManager(Config.CHECKPOINT_DIR, feature_type)
    
    # Load valid patients
    valid_patients = load_valid_patients(Config.VALID_PATIENTS_FILE)
    valid_slides, _ = get_valid_slides(Config.SVS_DIR, valid_patients)
    
    # Initialize DL extractor if needed
    dl_extractor = None
    if feature_type == 'dl':
        dl_extractor = LightweightDLExtractor(device=Config.DEVICE)
    
    # Filter out already processed slides
    remaining_slides = []
    for slide_path in valid_slides:
        slide_id = os.path.basename(slide_path).replace('.svs', '')
        if not checkpoint_mgr.is_slide_processed(slide_id):
            remaining_slides.append(slide_path)
    
    print(f"✓ Already processed: {len(valid_slides) - len(remaining_slides)} slides")
    print(f"✓ Remaining to process: {len(remaining_slides)} slides")
    
    if not remaining_slides:
        print("✓ All slides already processed!")
        return checkpoint_mgr.load_all_partial_features()
    
    # Process remaining slides
    start_time = time.time()
    success_count = 0
    fail_count = 0
    
    for idx, slide_path in enumerate(tqdm(remaining_slides, desc=f"Processing ({feature_type})")):
        slide_id = os.path.basename(slide_path).replace('.svs', '')
        
        success, num_features = process_slide_memory_efficient(
            slide_path, slide_id,
            patch_size=Config.PATCH_SIZE,
            n_patches=Config.PATCHES_PER_SLIDE,
            extraction_level=Config.EXTRACTION_LEVEL,
            checkpoint_mgr=checkpoint_mgr,
            feature_type=feature_type,
            dl_extractor=dl_extractor
        )
        
        if success:
            success_count += 1
        else:
            fail_count += 1
        
        # Force garbage collection
        gc.collect()
        
        # Print progress
        elapsed = time.time() - start_time
        print(f"\nProgress: {success_count} success, {fail_count} failed | "
              f"Time elapsed: {elapsed/60:.1f} min | "
              f"Avg per slide: {elapsed/(idx+1):.1f}s")
    
    # Load all features
    print(f"\n✓ Processing complete! Loading all features...")
    final_df = checkpoint_mgr.load_all_partial_features()
    
    # Save final output
    output_path = Config.OUTPUT_TRADITIONAL if feature_type == 'traditional' else Config.OUTPUT_DL
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY - {feature_type.upper()}")
    print(f"{'='*60}")
    print(f"✓ Slides processed: {success_count}/{len(remaining_slides)}")
    print(f"✓ Slides failed: {fail_count}")
    print(f"✓ Total feature vectors: {len(final_df)}")
    print(f"✓ Feature dimension: {len([c for c in final_df.columns if 'feat_' in c])}")
    print(f"✓ Output: {output_path}")
    print(f"✓ Total time: {(time.time() - start_time)/60:.1f} minutes")
    
    return final_df

# =============================================
# MAIN EXECUTION
# =============================================
if __name__ == "__main__":
    print("TCGA Integrative Analysis - Memory-Efficient Feature Extraction")
    print("="*60)
    print(f"Patch size: {Config.PATCH_SIZE}x{Config.PATCH_SIZE}")
    print(f"Patches per slide: {Config.PATCHES_PER_SLIDE}")
    print(f"Max patches in memory: {Config.MAX_PATCHES_IN_MEMORY}")
    print(f"Extraction level: {Config.EXTRACTION_LEVEL}")
    
    # Create output directories
    Path(Config.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    Path(Config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Phase 1: Traditional Features
    if Config.RUN_TRADITIONAL:
        traditional_df = run_processing_pipeline(feature_type='traditional')
    
    # Phase 2: Deep Learning Features
    if Config.RUN_DL:
        dl_df = run_processing_pipeline(feature_type='dl')
    
    print(f"\n{'#'*60}")
    print(f"ALL PROCESSING COMPLETE! 🎉")
    print(f"{'#'*60}")