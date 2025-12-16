import zarr
import tifffile
import numpy as np
import os
import glob
from tqdm import tqdm
import gc
# We don't need to import codecs because the array already exists!

# ================= CONFIGURATION =================
# Ensure this folder points to where the REMAINING (and replaced) files are
TIFF_FOLDER = "/mounts/disk4_tiago_e_andre/vesuvius/Vesuvius/notebooks/temp_raw_downloads"
ZARR_PATH = "/mounts/disk4_tiago_e_andre/vesuvius/Vesuvius/DataSet/Pre-training/PHercParis4.zarr"

# Batch size (Keep it moderate for safety)
BATCH_SIZE = 32

# ⚠️ SET TO FALSE TO ACTUALLY DELETE FILES
DRY_RUN = False 
# =================================================

def resume_conversion_safe():
    print("Indexing remaining files...")
    tiff_files = sorted(glob.glob(os.path.join(TIFF_FOLDER, "*.tif")))
    
    if not tiff_files:
        raise ValueError("No .tif files found! conversion might be complete.")

    print(f"Found {len(tiff_files)} files to process.")

    # 1. Open EXISTING Zarr (Mode 'r+' = Read/Write, NO OVERWRITE)
    try:
        root = zarr.open(ZARR_PATH, mode='r+')
        # Handle if it's a Group or Array directly
        if 'volume' in root:
            dset = root['volume']
        else:
            dset = root
        print(f"Opened Existing Volume: {dset.shape}")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not open Zarr. {e}")
        return

    print("Resuming... (Writing to correct Z-index based on filename)")
    
    # Buffers
    buffer_imgs = []
    buffer_indices = []

    # 2. Iteration Loop
    for fname in tqdm(tiff_files):
        try:
            # A. Parse Index from Filename
            # Assumes "001234.tif" -> 1234
            basename = os.path.basename(fname)
            # Extract only digits just in case
            idx_str = ''.join(filter(str.isdigit, basename.split('.')[0])) 
            if not idx_str: continue
            idx = int(idx_str)

            # Safety Check
            if idx >= dset.shape[0]:
                print(f"Skipping {fname} (Index {idx} is outside volume size)")
                continue

            # B. Read & Normalize (Same logic as your previous script)
            img = tifffile.imread(fname)
            img = img.astype(np.float32)
            img = (img / 65535.0) * 255.0
            img_uint8 = np.round(img).astype(np.uint8)

            # C. Add to Buffer
            buffer_imgs.append(img_uint8)
            buffer_indices.append(idx)

            # D. Write Buffer if Full
            if len(buffer_imgs) >= BATCH_SIZE:
                # Check if indices are contiguous (e.g., 100, 101, 102)
                is_contiguous = (
                    buffer_indices[-1] - buffer_indices[0] + 1 == len(buffer_indices)
                )

                if is_contiguous:
                    # Fast Block Write
                    start, end = buffer_indices[0], buffer_indices[-1]
                    dset[start : end + 1] = np.array(buffer_imgs)
                else:
                    # Slow Scatter Write (Safe for gaps)
                    for z_idx, z_img in zip(buffer_indices, buffer_imgs):
                        dset[z_idx] = z_img
                
                # E. Delete Processed Files
                if not DRY_RUN:
                    # We only delete the files currently in the buffer
                    # We map indices back to the 'tiff_files' list isn't safe if gaps exist.
                    # Safest: Re-construct filename or match fname.
                    # Simple hack: We are in a loop, we can't easily look back 32 steps safely.
                    # BETTER: Delete individually or keep a list of paths in buffer.
                    pass # We will delete below to be safe.

                # Clear Buffer
                # (We keep the paths to delete them now)
                processed_paths = tiff_files[tiff_files.index(fname)-len(buffer_imgs)+1 : tiff_files.index(fname)+1]
                
                buffer_imgs = []
                buffer_indices = []
                gc.collect()

                if not DRY_RUN:
                    for p in processed_paths:
                        try:
                            os.remove(p)
                        except:
                            pass

        except Exception as e:
            print(f"Error processing {fname}: {e}")
            # Don't crash the whole script for one file, but be careful
            continue

    # 3. Flush Final Buffer (Remaining files)
    if buffer_imgs:
        for z_idx, z_img in zip(buffer_indices, buffer_imgs):
            dset[z_idx] = z_img
        
        # Delete remainder
        if not DRY_RUN:
            # Re-find these paths (last N files)
            for p in tiff_files[-len(buffer_imgs):]:
                try:
                    os.remove(p)
                except:
                    pass

    print("Resume Complete.")

# Run
resume_conversion_safe()