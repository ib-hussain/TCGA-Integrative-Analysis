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
from pathlib import Path
import time
import traceback
import subprocess
import sys
warnings.filterwarnings('ignore')

# =============================================
# CONFIGURATION
# =============================================
class Config:
    # Paths
    SVS_DIR = "images/"
    VALID_PATIENTS_FILE = "csv/common_82_samples.txt"
    CHECKPOINT_DIR = "checkpoints/"
    OUTPUT_TRADITIONAL = "image_features_traditional.csv"
    OUTPUT_DL = "image_features_deeplearning.csv"
    
    # Patch extraction
    PATCH_SIZE = 256
    PATCHES_PER_SLIDE = 1000
    TISSUE_THRESHOLD = 0.3
    DETECTION_LEVEL = 5
    
    # Memory management
    MAX_PATCHES_MEMORY = 50   # Process this many patches at a time (lower = less peak RAM)
    SLIDES_BEFORE_SAVE = 5    # Save progress every N slides
    
    # Smoke test (set to True to test on just 2 slides with 50 patches each)
    SMOKE_TEST = False
    SMOKE_SLIDES = 2
    SMOKE_PATCHES = 50
    
    # Deep learning
    BATCH_SIZE = 32
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Hyperparameters for ResNet
    DL_MODEL = 'resnet50'  # 'resnet18' (lighter) or 'resnet50' (heavier)
    DL_IMAGE_SIZE = 224    # Input size for DL model

# =============================================
# CHECKPOINT MANAGEMENT
# =============================================
class CheckpointManager:
    """Manages checkpointing for resumable processing"""
    
    def __init__(self, checkpoint_dir, feature_type):
        self.checkpoint_dir = Path(checkpoint_dir) / feature_type
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.checkpoint_dir / "state.json"
        self.features_dir = self.checkpoint_dir / "features"
        self.features_dir.mkdir(exist_ok=True)
        
        # Load or initialize state
        self.state = self.load_state()
    
    def load_state(self):
        """Load processing state"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            'completed_slides': [],
            'failed_slides': {},
            'total_patches': 0,
            'current_slide_index': -1
        }
    
    def save_state(self):
        """Save processing state"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def is_slide_done(self, slide_id):
        """Check if slide is already processed"""
        return slide_id in self.state['completed_slides']
    
    def is_slide_failed(self, slide_id):
        """Check if slide previously failed"""
        return slide_id in self.state['failed_slides']
    
    def mark_slide_done(self, slide_id, num_patches):
        """Mark slide as completed"""
        if slide_id not in self.state['completed_slides']:
            self.state['completed_slides'].append(slide_id)
        self.state['total_patches'] += num_patches
        self.save_state()
    
    def mark_slide_failed(self, slide_id, error_msg):
        """Mark slide as failed"""
        self.state['failed_slides'][slide_id] = error_msg
        self.save_state()
    
    def save_slide_features(self, slide_id, df):
        """Save features for one slide"""
        filepath = self.features_dir / f"{slide_id}.csv"
        df.to_csv(filepath, index=False)
    
    def load_all_features(self):
        """Load all saved features"""
        files = list(self.features_dir.glob("*.csv"))
        if not files:
            return pd.DataFrame()
        
        dfs = []
        for f in tqdm(files, desc="Loading features"):
            try:
                dfs.append(pd.read_csv(f))
            except Exception as e:
                print(f"  [WARN] Error loading {f.name}: {e}")
        
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# =============================================
# SLIDE LOADING & VALIDATION
# =============================================
def load_valid_patients(valid_patients_file):
    """Load list of valid patient IDs"""
    with open(valid_patients_file, 'r') as f:
        valid_patients = set(line.strip() for line in f if line.strip())
    print(f"[OK] Loaded {len(valid_patients)} valid patient IDs")
    return valid_patients

def get_valid_slides(svs_dir, valid_patients):
    """Get SVS files that match valid patients"""
    all_svs = sorted(glob.glob(os.path.join(svs_dir, "*.svs")))
    valid_slides = []
    skipped = []
    
    for svs_path in all_svs:
        slide_id = os.path.basename(svs_path).replace('.svs', '')
        patient_id = slide_id[:12]
        
        if patient_id in valid_patients:
            valid_slides.append(svs_path)
        else:
            skipped.append(slide_id)
    
    print(f"[OK] Found {len(valid_slides)} valid slides out of {len(all_svs)} total")
    print(f"  Skipped {len(skipped)} slides (patients without complete omics data)")
    return valid_slides, skipped

# =============================================
# TISSUE DETECTION
# =============================================
def detect_tissue_mask(slide, level=None):
    """Create tissue mask at appropriate level"""
    if level is None:
        level = min(slide.level_count - 4, slide.level_count - 1)
    
    level = max(0, min(level, slide.level_count - 1))
    
    dims = slide.level_dimensions[level]
    img = slide.read_region((0, 0), level, dims)
    img_rgb = np.array(img.convert('RGB'))
    
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    
    _, mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    full_w, full_h = slide.dimensions
    mask_h, mask_w = mask.shape
    scale_x = full_w / mask_w
    scale_y = full_h / mask_h
    
    return mask, scale_x, scale_y

# =============================================
# PATCH EXTRACTION (MEMORY-EFFICIENT GENERATOR)
# =============================================
def extract_patches_generator(slide, patch_size=256, n_patches=500, tissue_threshold=0.3):
    """
    Generator that yields patches one at a time to save memory.
    Same logic as original extract_patches_smart but yields instead of storing.
    """
    mask, scale_x, scale_y = detect_tissue_mask(slide)
    
    full_w, full_h = slide.dimensions
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return
    
    total_tissue_area = sum(cv2.contourArea(c) for c in contours)
    if total_tissue_area == 0:
        return
    
    # Distribute patches proportionally
    tissues_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        n_region = max(1, int(n_patches * area / total_tissue_area))
        tissues_regions.append((contour, area, n_region))
    
    patches_yielded = 0
    
    for contour, area, n_region in tissues_regions:
        if patches_yielded >= n_patches:
            break
            
        x, y, w, h = cv2.boundingRect(contour)
        
        x_full = int(x * scale_x)
        y_full = int(y * scale_y)
        w_full = int(w * scale_x)
        h_full = int(h * scale_y)
        
        # Generate candidate positions
        step = patch_size // 2
        candidates_x = list(range(max(0, x_full), min(full_w - patch_size, x_full + w_full), step))
        candidates_y = list(range(max(0, y_full), min(full_h - patch_size, y_full + h_full), step))
        
        if not candidates_x or not candidates_y:
            continue
        
        n_candidates = len(candidates_x) * len(candidates_y)
        n_sample = min(n_region, n_candidates)
        
        if n_candidates == 0 or n_sample == 0:
            continue
        
        # Random sampling
        np.random.seed(None)
        indices = np.random.choice(n_candidates, size=min(n_sample, n_candidates), replace=False)
        
        for idx in indices:
            if patches_yielded >= n_patches:
                break
                
            ix = idx % len(candidates_x)
            iy = idx // len(candidates_x)
            
            px = candidates_x[ix]
            py = candidates_y[iy]
            
            if px < 0 or py < 0 or px + patch_size > full_w or py + patch_size > full_h:
                continue
            
            try:
                patch = slide.read_region((px, py), 0, (patch_size, patch_size))
                patch_rgb = np.array(patch.convert('RGB'))
                
                # Quick tissue check
                if np.mean(patch_rgb) < 240 and np.std(patch_rgb) > 10:
                    patches_yielded += 1
                    yield patch_rgb, (px, py)
            except:
                continue
    
    # If we didn't get enough patches, try random positions
    if patches_yielded < n_patches:
        attempts = 0
        while patches_yielded < n_patches and attempts < (n_patches - patches_yielded) * 5:
            px = np.random.randint(0, max(1, full_w - patch_size))
            py = np.random.randint(0, max(1, full_h - patch_size))
            
            # Check tissue mask
            mx = int(px / scale_x)
            my = int(py / scale_y)
            mw = int(patch_size / scale_x)
            mh = int(patch_size / scale_y)
            
            if (my + mh < mask.shape[0] and mx + mw < mask.shape[1]):
                mask_region = mask[my:my+mh, mx:mx+mw]
                tissue_percent = np.mean(mask_region) / 255.0
                
                if tissue_percent >= tissue_threshold:
                    try:
                        patch = slide.read_region((px, py), 0, (patch_size, patch_size))
                        patch_rgb = np.array(patch.convert('RGB'))
                        patches_yielded += 1
                        yield patch_rgb, (px, py)
                    except:
                        pass
            
            attempts += 1

# =============================================
# TRADITIONAL FEATURE EXTRACTION
# =============================================
def extract_traditional_features(patch):
    """Extract comprehensive traditional computer vision features"""
    try:
        gray = rgb2gray(patch)
        gray_uint = (gray * 255).astype(np.uint8)
        
        features = []
        
        # 1. Color features (15 features)
        for channel in range(3):
            ch = patch[:, :, channel]
            features.extend([
                np.mean(ch), np.std(ch), np.percentile(ch, 25),
                np.percentile(ch, 75), np.percentile(ch, 90)
            ])
        
        # 2. Texture - LBP (10 features)
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
        features.extend(lbp_hist)
        
        # 3. Texture - GLCM (10 features)
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
        
        # 4. Edge features (1 feature)
        edges = cv2.Canny(gray_uint, 50, 150)
        features.append(np.mean(edges) / 255)
        
        # 5. Nuclei-like features (4 features)
        hematoxylin = 1.0 * patch[:, :, 2] - 0.5 * patch[:, :, 0] - 0.5 * patch[:, :, 1]
        features.extend([np.mean(hematoxylin), np.std(hematoxylin)])
        
        eosin = 1.0 * patch[:, :, 0] + 0.5 * patch[:, :, 1] - 1.0 * patch[:, :, 2]
        features.extend([np.mean(eosin), np.std(eosin)])
        
        return np.array(features, dtype=np.float32)
    
    except Exception:
        return np.zeros(40, dtype=np.float32)

# =============================================
# DEEP LEARNING FEATURE EXTRACTOR
# =============================================
class DLFeatureExtractor:
    """Efficient batch-based deep learning feature extractor"""
    
    def __init__(self, model_name='resnet18', device='cpu', image_size=224):
        self.device = device
        self.image_size = image_size
        
        # Load model based on config
        weights_map = {
            'resnet18': ('IMAGENET1K_V1', 512),
            'resnet50': ('IMAGENET1K_V2', 2048),
        }
        
        weights, self.feature_dim = weights_map.get(model_name, weights_map['resnet18'])
        
        if model_name == 'resnet18':
            model = models.resnet18(weights=weights)
        else:
            model = models.resnet50(weights=weights)
        
        self.model = torch.nn.Sequential(*list(model.children())[:-1])
        self.model = self.model.to(device)
        self.model.eval()
        
        self.preprocess = transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"[OK] DL Extractor: {model_name} on {device} (dim={self.feature_dim})")
    
    def extract_batch(self, patches):
        """Extract features from a batch of patches"""
        batch_tensors = []
        
        for patch in patches:
            try:
                patch_pil = Image.fromarray(patch.astype(np.uint8))
                tensor = self.preprocess(patch_pil)
                batch_tensors.append(tensor)
            except:
                batch_tensors.append(torch.zeros(3, self.image_size, self.image_size))
        
        if not batch_tensors:
            return np.array([])
        
        batch = torch.stack(batch_tensors).to(self.device)
        
        with torch.no_grad():
            features = self.model(batch)
            features = features.squeeze(-1).squeeze(-1)
            result = features.cpu().numpy()
        
        del batch, batch_tensors, features
        if self.device != 'cpu':
            torch.cuda.empty_cache()
        return result

# =============================================
# SLIDE PROCESSOR (MEMORY-EFFICIENT)
# =============================================
def _flush_batch_to_csv(slide_id, features, coords_buffer, batch_num, filepath):
    """Write one batch of features directly to CSV (append mode). Returns row count."""
    rows = []
    for i, feat in enumerate(features):
        feature_dict = {
            'patient_id': slide_id[:12],
            'slide_id': slide_id,
            'tile_x': coords_buffer[i][0],
            'tile_y': coords_buffer[i][1],
            'tile_idx': batch_num * Config.MAX_PATCHES_MEMORY + i
        }
        for j, f in enumerate(feat):
            feature_dict[f'feat_{j:04d}'] = float(f)
        rows.append(feature_dict)
    
    df = pd.DataFrame(rows)
    write_header = not os.path.exists(filepath)
    df.to_csv(filepath, mode='a', header=write_header, index=False)
    n = len(df)
    del df, rows
    return n


def process_single_slide(svs_path, slide_id, feature_type, dl_extractor, n_patches, checkpoint_mgr):
    """Process one slide — streams each batch directly to disk to minimise RAM usage."""
    
    patches_buffer = []
    coords_buffer = []
    
    slide = None
    # Temporary per-slide CSV; moved into checkpoint dir on success
    tmp_path = checkpoint_mgr.features_dir / f"{slide_id}_tmp.csv"
    if tmp_path.exists():
        tmp_path.unlink()
    
    try:
        slide = openslide.OpenSlide(svs_path)
        
        patch_gen = extract_patches_generator(
            slide,
            patch_size=Config.PATCH_SIZE,
            n_patches=n_patches,
            tissue_threshold=Config.TISSUE_THRESHOLD
        )
        
        batch_num = 0
        patch_count = 0
        total_saved = 0
        
        for patch, coord in patch_gen:
            patches_buffer.append(patch)
            coords_buffer.append(coord)
            patch_count += 1
            
            if len(patches_buffer) >= Config.MAX_PATCHES_MEMORY:
                if feature_type == 'traditional':
                    features = [extract_traditional_features(p) for p in patches_buffer]
                else:
                    features = dl_extractor.extract_batch(patches_buffer)
                
                total_saved += _flush_batch_to_csv(
                    slide_id, features, coords_buffer, batch_num, tmp_path
                )
                batch_num += 1
                
                # Release everything immediately
                del features, patches_buffer, coords_buffer
                patches_buffer = []
                coords_buffer = []
                gc.collect()
                
                print(f"  [{slide_id}] Batch {batch_num}: {patch_count} patches processed", end='\r')
        
        # Flush remaining patches
        if patches_buffer:
            if feature_type == 'traditional':
                features = [extract_traditional_features(p) for p in patches_buffer]
            else:
                features = dl_extractor.extract_batch(patches_buffer)
            
            total_saved += _flush_batch_to_csv(
                slide_id, features, coords_buffer, batch_num, tmp_path
            )
            del features, patches_buffer, coords_buffer
            gc.collect()
        
        if total_saved > 0:
            # Rename tmp file to final checkpoint name
            final_path = checkpoint_mgr.features_dir / f"{slide_id}.csv"
            tmp_path.rename(final_path)
            checkpoint_mgr.mark_slide_done(slide_id, total_saved)
            return total_saved
        
        # Clean up empty tmp file
        if tmp_path.exists():
            tmp_path.unlink()
        return 0
    
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        checkpoint_mgr.mark_slide_failed(slide_id, f"{str(e)}\n{traceback.format_exc()}")
        raise
    
    finally:
        if slide:
            slide.close()
        gc.collect()


# =============================================
# SUBPROCESS WORKER ENTRY POINT
# =============================================
def _subprocess_worker(svs_path, slide_id, feature_type, n_patches, checkpoint_dir):
    """Entry point when called as subprocess for a single slide."""
    checkpoint_mgr = CheckpointManager(checkpoint_dir, feature_type)
    dl_extractor = None
    if feature_type == 'dl':
        dl_extractor = DLFeatureExtractor(
            model_name=Config.DL_MODEL,
            device=Config.DEVICE,
            image_size=Config.DL_IMAGE_SIZE
        )
    n = process_single_slide(svs_path, slide_id, feature_type, dl_extractor, n_patches, checkpoint_mgr)
    print(f"PATCHES:{n}")


def _run_slide_subprocess(svs_path, slide_id, feature_type, n_patches, checkpoint_mgr):
    """
    Spawn an isolated subprocess per slide so the OS fully reclaims its
    memory (OpenSlide cache, torch tensors, numpy buffers) after each slide.
    """
    cmd = [
        sys.executable, __file__,
        '--worker',
        '--svs', svs_path,
        '--slide-id', slide_id,
        '--feature-type', feature_type,
        '--n-patches', str(n_patches),
        '--checkpoint-dir', str(checkpoint_mgr.checkpoint_dir.parent),
    ]
    # Force UTF-8 in subprocess so Unicode checkmarks don't crash on Windows (cp1252)
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', env=env)

    if result.returncode != 0:
        err = result.stderr.strip().splitlines()
        raise RuntimeError(err[-1] if err else "subprocess failed with no stderr")

    for line in result.stdout.splitlines():
        if line.startswith("PATCHES:"):
            return int(line.split(":")[1])
    return 0


# =============================================
# MAIN PROCESSING PIPELINE
# =============================================
def run_pipeline(feature_type='traditional', smoke_test=False):
    """Main processing pipeline with checkpoint support"""
    
    print(f"\n{'='*60}")
    print(f"PROCESSING: {feature_type.upper()} FEATURES")
    print(f"{'='*60}")
    
    if smoke_test:
        print(f"*** SMOKE TEST MODE ***")
    
    # Setup checkpoint
    checkpoint_mgr = CheckpointManager(Config.CHECKPOINT_DIR, feature_type)
    
    # Load patients
    valid_patients = load_valid_patients(Config.VALID_PATIENTS_FILE)
    valid_slides, _ = get_valid_slides(Config.SVS_DIR, valid_patients)
    
    # Smoke test: only use first few slides
    if smoke_test:
        valid_slides = valid_slides[:Config.SMOKE_SLIDES]
        n_patches_per_slide = Config.SMOKE_PATCHES
    else:
        n_patches_per_slide = Config.PATCHES_PER_SLIDE
    
    # Filter already processed slides
    remaining_slides = []
    for sp in valid_slides:
        sid = os.path.basename(sp).replace('.svs', '')
        if not checkpoint_mgr.is_slide_done(sid):
            remaining_slides.append(sp)
    
    print(f"[OK] Already completed: {len(valid_slides) - len(remaining_slides)} slides")
    print(f"[OK] Remaining to process: {len(remaining_slides)} slides")
    print(f"[OK] Patches per slide: {n_patches_per_slide}")
    
    if not remaining_slides:
        print("[OK] All slides already processed!")
        return checkpoint_mgr.load_all_features()
    
    # Process slides (each slide runs in its own subprocess - dl_extractor init is handled per-subprocess)
    start_time = time.time()
    success_count = 0
    fail_count = 0
    
    for idx, svs_path in enumerate(tqdm(remaining_slides, desc=f"Processing slides ({feature_type})")):
        slide_id = os.path.basename(svs_path).replace('.svs', '')
        
        try:
            n_features = _run_slide_subprocess(
                svs_path, slide_id, feature_type,
                n_patches_per_slide, checkpoint_mgr
            )
            
            if n_features > 0:
                success_count += 1
                print(f"\n  [OK] [{slide_id}] Done: {n_features} patches")
            else:
                fail_count += 1
                print(f"\n  [WARN] [{slide_id}] No tissue found")
        
        except Exception as e:
            fail_count += 1
            print(f"\n  [FAIL] [{slide_id}] Failed: {str(e)[:100]}")
        
        # Progress update
        elapsed = time.time() - start_time
        rate = elapsed / (idx + 1)
        remaining = rate * (len(remaining_slides) - idx - 1)
        print(f"Progress: {success_count} ok, {fail_count} fail | "
              f"Elapsed: {elapsed/60:.1f}m | ETA: {remaining/60:.1f}m | "
              f"Rate: {rate:.1f}s/slide")
    
    # Load and save final output
    print("\n[OK] Loading all features...")
    final_df = checkpoint_mgr.load_all_features()
    
    output_file = Config.OUTPUT_TRADITIONAL if feature_type == 'traditional' else Config.OUTPUT_DL
    
    if len(final_df) > 0:
        final_df.to_csv(output_file, index=False)
        
        feature_cols = [c for c in final_df.columns if c.startswith('feat_')]
        
        print(f"\n{'='*60}")
        print(f"{feature_type.upper()} FEATURES EXTRACTION COMPLETE")
        print(f"{'='*60}")
        print(f"[OK] Successful slides: {success_count}/{len(remaining_slides)}")
        print(f"[OK] Total patches: {len(final_df)}")
        print(f"[OK] Feature dimension: {len(feature_cols)}")
        print(f"[OK] Output: {output_file}")
        print(f"[OK] Total time: {(time.time() - start_time)/60:.1f} minutes")
        print(f"[OK] Checkpoints: {Config.CHECKPOINT_DIR}/{feature_type}/")
    else:
        print("\n[FAIL] No features extracted!")
    
    return final_df

# =============================================
# MAIN EXECUTION
# =============================================
if __name__ == "__main__":
    import argparse
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument('--worker', action='store_true')
    _parser.add_argument('--svs', default=None)
    _parser.add_argument('--slide-id', default=None)
    _parser.add_argument('--feature-type', default=None)
    _parser.add_argument('--n-patches', type=int, default=None)
    _parser.add_argument('--checkpoint-dir', default=None)
    _args, _ = _parser.parse_known_args()

    if _args.worker:
        # Running as a subprocess worker for a single slide
        _subprocess_worker(
            svs_path=_args.svs,
            slide_id=_args.slide_id,
            feature_type=_args.feature_type,
            n_patches=_args.n_patches,
            checkpoint_dir=_args.checkpoint_dir,
        )
        sys.exit(0)

    print("TCGA Integrative Analysis - Image Feature Extraction")
    print("="*60)
    
    # Configuration
    print(f"\nConfiguration:")
    print(f"  Patch size: {Config.PATCH_SIZE}x{Config.PATCH_SIZE}")
    print(f"  Patches per slide: {Config.PATCHES_PER_SLIDE}")
    print(f"  Batch size (memory): {Config.MAX_PATCHES_MEMORY}")
    print(f"  Device: {Config.DEVICE}")
    print(f"  DL Model: {Config.DL_MODEL}")
    print(f"  Smoke Test: {Config.SMOKE_TEST}")
    
    # Create directories
    Path(Config.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    
    if Config.SMOKE_TEST:
        print(f"\n{'#'*60}")
        print(f"RUNNING SMOKE TEST FIRST")
        print(f"{'#'*60}")
        
        # Run smoke test for traditional features
        print("\n>>> Smoke Test: Traditional Features")
        try:
            _ = run_pipeline(feature_type='traditional', smoke_test=True)
            print("\n[OK] Traditional features smoke test PASSED")
        except Exception as e:
            print(f"\n[FAIL] Traditional features smoke test FAILED: {e}")
        
        # Run smoke test for DL features
        print("\n>>> Smoke Test: Deep Learning Features")
        try:
            _ = run_pipeline(feature_type='dl', smoke_test=True)
            print("\n[OK] DL features smoke test PASSED")
        except Exception as e:
            print(f"\n[FAIL] DL features smoke test FAILED: {e}")
        
        print(f"\n{'#'*60}")
        print(f"SMOKE TEST COMPLETE - Starting full pipeline")
        print(f"{'#'*60}")
    
    # Full pipeline
    Config.SMOKE_TEST = False  # Disable smoke test for full run
    
    print(f"\n{'#'*60}")
    print(f"FULL PIPELINE")
    print(f"{'#'*60}")
    
    # Phase 1: Traditional Features
    traditional_features = run_pipeline(feature_type='traditional')
    
    # Phase 2: Deep Learning Features
    dl_features = run_pipeline(feature_type='dl')
    
    print(f"\n{'#'*60}")
    print(f"ALL PROCESSING COMPLETE! ")
    print(f"{'#'*60}")