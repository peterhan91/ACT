import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import nibabel as nib
import h5py
import torch
import torch.nn.functional as F
from typing import *
from pathlib import Path

def load_data(filepath):
    dataframe = pd.read_csv(filepath)
    return dataframe

def get_ct_paths_list(filepath): 
    dataframe = load_data(filepath)
    ct_paths = dataframe['Path']
    return ct_paths

def read_and_preprocess_ct(file_path, metadata_row=None, hu_range=(-1000, 1000), return_spacing=False):
    """
    Load and preprocess a CT scan:
    - Reorient to RAS (Right-Anterior-Superior) orientation
    - Apply DICOM rescale (if metadata available)
    - Clip to [-1000, 1000] HU range
    - Optionally return voxel spacing information
    """
    img = nib.load(str(file_path))
    
    # Reorient to RAS orientation (Right-Anterior-Superior)
    # This ensures consistent orientation: left->right, posterior->anterior, inferior->superior
    img = nib.as_closest_canonical(img)
    
    data = img.get_fdata()
    
    # Get voxel spacing from header (in mm)
    # After transpose, spacing order becomes (z, y, x)
    header = img.header
    spacing = header.get_zooms()[:3]  # Get first 3 dimensions
    spacing_transposed = (spacing[2], spacing[1], spacing[0])  # Transpose to match data
    
    # Apply DICOM rescale if metadata is available
    if metadata_row is not None:
        slope = 1.0 # float(metadata_row['RescaleSlope'])
        intercept = 0.0 # float(metadata_row['RescaleIntercept'])
        data = data * slope + intercept
    else:
        # Use default rescale values: slope=1, intercept=0
        print(f"Warning: No metadata for {file_path}, using default rescale (slope=1, intercept=0)")
        data = data
    
    data = np.clip(data, hu_range[0], hu_range[1])
    data_transposed = data.transpose(2, 1, 0).astype(np.float32)
    
    if return_spacing:
        return data_transposed, spacing_transposed
    else:
        return data_transposed

def resample_to_isotropic_spacing(volume, original_spacing, target_in_plane_mm=1.5, target_out_plane_mm=3.0):
    """
    Resample CT volume to isotropic spacing:
    - In-plane (first two axes): 1.5mm resolution
    - Out-of-plane (last axis): 3mm slice thickness
    Uses bilinear interpolation as specified.
    
    Args:
        volume: numpy array of shape (D, H, W) after transpose in read_and_preprocess_ct
        original_spacing: tuple of (spacing_z, spacing_y, spacing_x) in mm
        target_in_plane_mm: target in-plane resolution in mm (default 1.5)
        target_out_plane_mm: target out-of-plane slice thickness in mm (default 3.0)
    
    Returns:
        resampled volume as numpy array
    """
    d, h, w = volume.shape
    spacing_z, spacing_y, spacing_x = original_spacing
    
    # Calculate new dimensions based on spacing
    new_d = int(round(d * spacing_z / target_out_plane_mm))
    new_h = int(round(h * spacing_y / target_in_plane_mm))
    new_w = int(round(w * spacing_x / target_in_plane_mm))
    
    # Resample using bilinear interpolation (mode='bilinear' for 2D slices, 'trilinear' for 3D)
    tensor = torch.tensor(volume).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    tensor_resampled = F.interpolate(
        tensor, size=(new_d, new_h, new_w), mode='trilinear', align_corners=False
    )
    resampled = tensor_resampled.squeeze().cpu().numpy()
    
    return resampled

def resize_and_pad_3d_to_target(volume, target_shape=(160, 224, 224), slide_b2u=True):
    """
    Aspect-ratio-preserving resize to fit within target_shape, then center-pad to match it.
    Normalize HU values to [-1, 1] range, then convert to uint8 [0, 255].
    """
    d, h, w = volume.shape
    target_d, target_h, target_w = target_shape
    scale = min(target_d / d, target_h / h, target_w / w)
    new_d, new_h, new_w = [int(round(x * scale)) for x in (d, h, w)]

    # Resize (trilinear) - keeping HU values
    tensor = torch.tensor(volume).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    tensor_resized = F.interpolate(
        tensor, size=(new_d, new_h, new_w), mode='trilinear', align_corners=False
    )
    resized = tensor_resized.squeeze().cpu().numpy()

    # Center padding with 0 HU (water density)
    pad_d = target_d - new_d
    pad_h = target_h - new_h
    pad_w = target_w - new_w
    pad = (
        pad_w // 2, pad_w - pad_w // 2,
        pad_h // 2, pad_h - pad_h // 2,
        pad_d // 2, pad_d - pad_d // 2,
    )
    padded = np.pad(
        resized,
        ((pad[4], pad[5]), (pad[2], pad[3]), (pad[0], pad[1])),
        mode='constant',
        constant_values=-1000
    )
    
    # Normalize to [-1, 1] then convert to [0, 255] for uint8
    normalized = padded / 1000.0  # [-1, 1]
    uint8_data = ((normalized + 1.0) / 2.0 * 255.0)  # Map [-1,1] to [0,255]
    uint8_data = np.clip(uint8_data, 0, 255)
    if slide_b2u:
        return uint8_data.astype(np.uint8)
    else:
        return uint8_data[:, ::-1, :].astype(np.uint8)

def resize_and_pad_3d_isotropic(volume, target_shape=(160, 224, 224), slide_b2u=True):
    """
    Process isotropically resampled volume: center crop/pad to target shape.
    Normalize HU values to [-1, 1] range, then convert to uint8 [0, 255].
    """
    d, h, w = volume.shape
    target_d, target_h, target_w = target_shape
    
    # Center crop or pad to target dimensions
    def center_crop_or_pad_1d(size, target_size):
        if size >= target_size:
            # Center crop
            start = (size - target_size) // 2
            return start, start + target_size
        else:
            # Will need padding
            return 0, size
    
    # Calculate crop/pad for each dimension
    d_start, d_end = center_crop_or_pad_1d(d, target_d)
    h_start, h_end = center_crop_or_pad_1d(h, target_h)
    w_start, w_end = center_crop_or_pad_1d(w, target_w)
    
    # Apply cropping first
    cropped = volume[d_start:d_end, h_start:h_end, w_start:w_end]
    
    # Calculate padding needed
    cd, ch, cw = cropped.shape
    pad_d = target_d - cd
    pad_h = target_h - ch
    pad_w = target_w - cw
    
    # Apply padding if needed
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        pad = (
            pad_w // 2, pad_w - pad_w // 2,
            pad_h // 2, pad_h - pad_h // 2,
            pad_d // 2, pad_d - pad_d // 2,
        )
        processed = np.pad(
            cropped,
            ((pad[4], pad[5]), (pad[2], pad[3]), (pad[0], pad[1])),
            mode='constant',
            constant_values=-1000
        )
    else:
        processed = cropped
    
    # Normalize to [-1, 1] then convert to [0, 255] for uint8
    normalized = processed / 1000.0  # [-1, 1]
    uint8_data = ((normalized + 1.0) / 2.0 * 255.0)  # Map [-1,1] to [0,255]
    uint8_data = np.clip(uint8_data, 0, 255)
    if slide_b2u:
        return uint8_data.astype(np.uint8)
    else:
        return uint8_data[:, ::-1, :].astype(np.uint8)

def process_single_ct(args):
    """
    Helper function for parallel processing of a single CT volume.
    Returns (idx, processed_volume, filename) or (idx, None, filename) on failure.
    """
    idx, path, metadata_df, target_shape, slide_b2u, iso_spacing = args
    try:
        fname = Path(path).name
        metadata_row = None
        
        # Try to find metadata if available
        if metadata_df is not None:
            row = metadata_df[metadata_df['VolumeName'] == fname]
            if not row.empty:
                metadata_row = row.iloc[0]
            else:
                print(f"Metadata missing for: {fname}, using defaults")
        else:
            print(f"No metadata available for: {fname}, using defaults")
            
        # Read and preprocess CT
        if iso_spacing:
            # For isotropic spacing, we need the original spacing information
            vol, spacing = read_and_preprocess_ct(path, metadata_row, return_spacing=True)
            # Resample to isotropic spacing
            vol = resample_to_isotropic_spacing(vol, spacing)
            # Use isotropic processing function
            vol = resize_and_pad_3d_isotropic(vol, target_shape=target_shape, slide_b2u=slide_b2u)
        else:
            # Use original ratio-preserving processing
            vol = read_and_preprocess_ct(path, metadata_row)
            vol = resize_and_pad_3d_to_target(vol, target_shape=target_shape, slide_b2u=slide_b2u)
        
        return (idx, vol, fname, None)
        
    except Exception as e:
        return (idx, None, Path(path).name, str(e))

def ct_to_hdf5(ct_paths: List[Union[str, Path]], metadata_df: pd.DataFrame = None, out_filepath: str = "ct_volumes.h5", target_shape=(160, 224, 224), num_workers=4, slide_b2u=True, iso_spacing=False): 
    """
    Convert directory of CT scans into a .h5 file given paths to all 
    CT volumes and their metadata. Uses streaming writes to avoid OOM.
    """
    dset_size = len(ct_paths)
    failed_volumes = []
    
    # Modify output filename for isotropic spacing
    if iso_spacing and not out_filepath.endswith('_iso_spacing.h5'):
        base_name = out_filepath.replace('.h5', '')
        out_filepath = f"{base_name}_iso_spacing.h5"
    
    with h5py.File(out_filepath, 'w') as h5f:
        ct_dset = h5f.create_dataset(
            'ct_volumes', 
            shape=(dset_size, *target_shape), 
            dtype='uint8',
        )
        
        # Process in batches - scale with available memory
        batch_size = min(num_workers * 2, 32)  # Allow larger batches with more memory
        processing_type = "isotropic spacing" if iso_spacing else "ratio-preserving"
        print(f"Processing {len(ct_paths)} CT volumes with {processing_type} in batches of {batch_size}...")
        
        for batch_start in tqdm(range(0, dset_size, batch_size), desc="Batches"):
            batch_end = min(batch_start + batch_size, dset_size)
            batch_paths = ct_paths[batch_start:batch_end]
            
            # Prepare arguments for this batch
            process_args = [(idx + batch_start, path, metadata_df, target_shape, slide_b2u, iso_spacing) 
                          for idx, path in enumerate(batch_paths)]
            
            # Process batch in parallel
            batch_results = [None] * len(batch_paths)
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_idx = {executor.submit(process_single_ct, arg): arg[0] - batch_start 
                               for arg in process_args}
                
                for future in as_completed(future_to_idx):
                    batch_idx = future_to_idx[future]
                    global_idx, vol, fname, error = future.result()
                    
                    if error is not None:
                        failed_volumes.append((ct_paths[global_idx], error))
                        batch_results[batch_idx] = None
                    else:
                        batch_results[batch_idx] = (global_idx, vol, fname)
            
            # Write batch results immediately to HDF5
            for batch_idx, result in enumerate(batch_results):
                if result is not None:
                    global_idx, vol, fname = result
                    ct_dset[global_idx] = vol
                    # Explicitly delete volume from memory
                    del vol
                # Set result to None to free memory immediately
                batch_results[batch_idx] = None
            
            # Clear batch results to free memory
            del batch_results
            
            # Force garbage collection after each batch
            import gc
            gc.collect()
                
    print(f"{len(failed_volumes)} / {len(ct_paths)} volumes failed to be added to h5.")
    if failed_volumes:
        print("Failed volumes:", failed_volumes[:10])  # Show first 10 failures

def get_files(directory):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(directory):
        for file in filenames:
            if file.endswith(".nii.gz"):
                files.append(os.path.join(dirpath, file))
    return files

def get_ct_path_csv(out_filepath, directory):
    files = get_files(directory)
    file_dict = {"Path": files}
    df = pd.DataFrame(file_dict)
    df.to_csv(out_filepath, index=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_out_path', type=str, default='data/ct_paths.csv', help="Directory to save paths to all CT volumes in dataset.")
    parser.add_argument('--ct_out_path', type=str, default='data/ct_volumes.h5', help="Directory to save processed CT volume data.")
    parser.add_argument('--metadata_path', type=str, default=None, help="Path to metadata CSV file containing rescale slope/intercept (optional).")
    parser.add_argument('--ct_data_path', default='data/ct_volumes/', help="Directory where CT volume data is stored.")
    parser.add_argument('--target_shape', nargs=3, type=int, default=[160, 224, 224], help="Target shape for CT volumes (D H W).")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of worker threads for parallel processing.")
    parser.add_argument('--slide_b2u', type=bool, default=False, help="Slice ordering from bottom to up (default: True).")
    parser.add_argument('--split', type=str, choices=['train', 'valid', 'test'], help="Process specific split using pre-generated CSV paths.")
    parser.add_argument('--dataset', type=str, choices=['ctrate', 'inspect', 'merlin'], default='ctrate', help="Dataset type: ctrate, inspect, or merlin (default: ctrate).")
    parser.add_argument('--iso_spacing', action='store_true', help="Enable isotropic spacing processing: resample to 1.5mm in-plane, 3mm out-of-plane. Output file will have '_iso_spacing' suffix.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    # Load metadata if provided
    metadata_df = None
    if args.metadata_path and os.path.exists(args.metadata_path):
        print(f"Loading metadata from: {args.metadata_path}")
        metadata_df = pd.read_csv(args.metadata_path)
        print(f"Loaded metadata for {len(metadata_df)} volumes")
    else:
        if args.metadata_path:
            print(f"Warning: Metadata file not found: {args.metadata_path}")
        print("Processing without metadata - using default rescale values")
    
    # Get CT paths
    if args.split:
        # Use pre-generated CSV paths for specific split
        csv_path = f"/path/to/ACT/preprocessing/run_scripts/{args.dataset}_{args.split}_paths.csv"
        if os.path.exists(csv_path):
            print(f"Using pre-generated paths from: {csv_path}")
            ct_paths = get_ct_paths_list(csv_path)
        else:
            raise FileNotFoundError(f"Split CSV not found: {csv_path}. Run generate_split_csvs.py (for ctrate), generate_inspect_split_csvs.py (for inspect), or generate_merlin_split_csvs.py (for merlin) first.")
    elif args.ct_data_path.endswith('.csv'):
        # Handle CSV file directly provided in --ct_data_path
        print(f"Using CSV file provided in --ct_data_path: {args.ct_data_path}")
        ct_paths = get_ct_paths_list(args.ct_data_path)
    else:
        # Legacy mode: scan directory and create CSV
        get_ct_path_csv(args.csv_out_path, args.ct_data_path)
        ct_paths = get_ct_paths_list(args.csv_out_path)
    
    # Modify output path to include dataset name if using split mode
    output_path = args.ct_out_path
    if args.split and args.dataset != 'ctrate':
        # Add dataset prefix to output file name only if not already present
        base_name = os.path.splitext(output_path)[0]
        extension = os.path.splitext(output_path)[1]
        # Check if dataset name is already in the filename
        if args.dataset not in base_name:
            output_path = f"{base_name}_{args.dataset}{extension}"
    
    print(f"Processing {len(ct_paths)} CT volumes...")
    print(f"Output will be saved to: {output_path}")
    ct_to_hdf5(ct_paths, metadata_df, output_path, target_shape=tuple(args.target_shape), num_workers=args.num_workers, slide_b2u=args.slide_b2u, iso_spacing=args.iso_spacing)