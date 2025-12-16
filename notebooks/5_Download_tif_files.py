import os
import requests
import numpy as np
import tifffile
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import urljoin

# --- CONFIGURATION ---

"""
# List of available scrolls:
['https://dl.ash2txt.org/full-scrolls/Scroll2/PHercParis3.volpkg/volumes_masked/20230210143520/', 
'https://dl.ash2txt.org/full-scrolls/Scroll2/PHercParis3.volpkg/volumes_masked/20230212125146/',
''
]
"""
TARGET_URL = "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_masked/20230205180739/"
DOWNLOAD_DIR = "./temp_raw_downloads"
OUTPUT_DIR = "../DataSet/Pre-training/cropped_output"

def get_tif_links(url):
    """Scrapes the directory for .tif files."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        links = [a['href'] for a in soup.find_all('a') if a['href'].endswith('.tif')]
        return sorted(links)
    except Exception as e:
        print(f"Error fetching links: {e}")
        return []
import os
import time
import requests

def download_file(file_url, local_path, max_retries=10, timeout=30):
    """Downloads a single file with retries and backoff, skipping if it already exists."""
    
    # Skip if file already exists and is non-empty
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return

    for attempt in range(max_retries):
        try:
            with requests.get(file_url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                
                # Create parent folder if needed
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                with open(local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:  # skip keep-alive chunks
                            f.write(chunk)

            # Success → return
            return

        except Exception as e:
            wait = 2 ** attempt
            print(f"[Retry {attempt+1}/{max_retries}] Error downloading {file_url}: {e}")
            print(f"Waiting {wait}s before retry...")
            time.sleep(wait)

    raise RuntimeError(f"Failed to download after {max_retries} retries: {file_url}")


import tifffile
import numpy as np

def crop_foreground(tiff_path, bounds):

    x_min = bounds['x_min']
    x_max = bounds['x_max']
    y_min = bounds['y_min']
    y_max = bounds['y_max']
    
    # Load image
    tiff_img = tifffile.imread(tiff_path)



    # Crop
    cropped = tiff_img[y_min:y_max+1, x_min:x_max+1]
    return cropped

def get_small_sample_list(tif_files):
    # Suppose tif_files is your list of file paths
    n_samples = 100

    # Handle case when there are fewer than 100 files
    n_samples = min(n_samples, len(tif_files))

    # Get 100 equally spaced indices
    indices = np.linspace(0, len(tif_files)-1, n_samples, dtype=int)

    # Select files
    selected_files = [tif_files[i] for i in indices]
    return selected_files

def calculate_global_bounds(file_list):
    """
    Iterates through 2D files to calculate the 3D bounding box 
    without loading the whole volume into RAM.
    Implements your 'detect_deges_with_content' logic iteratively.
    """
    print("\n--- Scanning files to calculate Global Bounding Box ---")
    
    # Initialize with inverted infinity to find min/max
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    min_z, max_z = float('inf'), float('-inf')
    
    has_data = False

    for z_index, filename in enumerate(tqdm(file_list, desc="Scanning Geometry")):
        file_path = os.path.join(DOWNLOAD_DIR, filename)
        
        # Load 2D slice
        img = tifffile.imread(file_path)
        
        # Check if slice has any data (Your Z-axis check)
        if np.any(img > 0):
            # Update Z bounds
            if z_index < min_z: min_z = z_index
            if z_index > max_z: max_z = z_index
            
            has_data = True
            
            # Check Y axis (Rows) for this slice
            # np.any(img, axis=1) gives boolean array of rows containing data
            y_indices = np.where(np.any(img > 0, axis=1))[0]
            if len(y_indices) > 0:
                current_y_min, current_y_max = y_indices[0], y_indices[-1]
                min_y = min(min_y, current_y_min)
                max_y = max(max_y, current_y_max)

            # Check X axis (Cols) for this slice
            # np.any(img, axis=0) gives boolean array of cols containing data
            x_indices = np.where(np.any(img > 0, axis=0))[0]
            if len(x_indices) > 0:
                current_x_min, current_x_max = x_indices[0], x_indices[-1]
                min_x = min(min_x, current_x_min)
                max_x = max(max_x, current_x_max)

    if not has_data:
        return None

    # Convert to integers
    bounds = {
        'x_min': int(min_x), 'x_max': int(max_x),
        'y_min': int(min_y), 'y_max': int(max_y),
        'z_min': int(min_z), 'z_max': int(max_z)
    }
    return bounds

# 1. Setup
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# 2. Get File List
print("Fetching file list...")
tif_files = get_tif_links(TARGET_URL)
if not tif_files:
    print("No files found.")
    exit()
    
print(f"Found {len(tif_files)} files.")

# 3. Download (One by one), crop and store locally
# We will divide by chunks of x, y, 1024 -> so we can still have volumes, a good representation of data and save disk space
print("Downloading files...")

chunk_size = 1024

chunk_list = [tif_files[i:i + chunk_size] for i in range(0, len(tif_files), chunk_size)]

for chunk_idx, tif_files in enumerate(chunk_list):
    print(f"\nProcessing chunk {chunk_idx+1}/{len(chunk_list)}...")
    OUTPUT_DIR_chunk = os.path.join(OUTPUT_DIR, f"chunk_{chunk_idx}")
    os.makedirs(OUTPUT_DIR_chunk, exist_ok=True)
    
    for idx_z, f in enumerate(tqdm(tif_files, desc="Downloading")):
        url = urljoin(TARGET_URL, f)
        path = os.path.join(DOWNLOAD_DIR, f)
        download_file(url, path)
  
    # Calculate bounds
    #bounds = calculate_global_bounds(tif_files)
    #print(f"Current bounds at slice {idx_z}: {bounds}")
    
    #for idx_z, f in enumerate(tqdm(tif_files, desc="Croping and Saving")):
        # crop edges without content, save and remove tmp file
        #path = os.path.join(DOWNLOAD_DIR, f)
        #cropped_numpy = crop_foreground(tiff_path=path, bounds=bounds)
        #output_path = os.path.join(OUTPUT_DIR_chunk, f)
        #tifffile.imwrite(output_path, cropped_numpy, compression='deflate')
        #os.remove(path)
