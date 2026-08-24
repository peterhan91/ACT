#!/usr/bin/env python3
"""
Generate filtered path CSV files for each CT-RATE split.
This creates ctrate_train_paths.csv, ctrate_valid_paths.csv, and ctrate_test_paths.csv
containing the full paths to the NII files for each split.
"""

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def find_volume_path(volume_name, search_dirs):
    """Find the full path of a volume using hardcoded patterns."""
    # Extract pattern from volume name (e.g., train_1_a_1.nii.gz -> train_1/train_1_a/)
    parts = volume_name.replace('.nii.gz', '').split('_')
    if len(parts) >= 3:
        # Pattern: {split}_{id}_{sub}_{num}.nii.gz -> {split}_fixed/{split}_{id}/{split}_{id}_{sub}/
        split_type = parts[0]  # train or valid
        patient_id = parts[1]  # numeric ID
        sub_id = parts[2]      # a, b, etc.
        
        patient_dir = f"{split_type}_{patient_id}"
        sub_dir = f"{split_type}_{patient_id}_{sub_id}"
        
        # Try both search directories
        for search_dir in search_dirs:
            pattern_path = os.path.join(search_dir, patient_dir, sub_dir, volume_name)
            if os.path.exists(pattern_path):
                return pattern_path
    
    return None

def create_split_csv(split_name, metadata_csv, search_dirs, output_csv):
    """Create a CSV with full paths for a specific split."""
    print(f"\n=== Processing {split_name.upper()} split ===")
    
    # Load metadata
    metadata_df = pd.read_csv(metadata_csv)
    volume_names = metadata_df['VolumeName'].tolist()
    print(f"Looking for {len(volume_names)} volumes...")
    
    # Find paths
    paths = []
    missing = []
    
    for volume_name in tqdm(volume_names, desc="Finding paths"):
        full_path = find_volume_path(volume_name, search_dirs)
        if full_path:
            paths.append(full_path)
        else:
            missing.append(volume_name)
    
    # Save to CSV
    paths_df = pd.DataFrame({'Path': paths})
    paths_df.to_csv(output_csv, index=False)
    
    print(f"✅ Created {output_csv}")
    print(f"   Found: {len(paths)} volumes")
    if missing:
        print(f"   Missing: {len(missing)} volumes")
        if len(missing) <= 5:
            print(f"   Missing files: {missing}")
        else:
            print(f"   First 5 missing: {missing[:5]}")
    
    return len(paths), len(missing)

def main():
    # Paths
    metadata_dir = "/path/to/ACT/model/data/ct_rate"
    data_base_dir = "/path/to/data/CT_RATE_v2.0/dataset"
    output_dir = "/path/to/ACT/preprocessing/run_scripts"
    
    # Search directories
    search_dirs = [
        os.path.join(data_base_dir, "train_fixed"),
        os.path.join(data_base_dir, "valid_fixed")
    ]
    
    print("CT-RATE Split CSV Generator")
    print(f"Data directories: {search_dirs}")
    print(f"Output directory: {output_dir}")
    
    # Define splits
    splits = {
        'train': {
            'metadata': os.path.join(metadata_dir, "train_metadata.csv"),
            'output': os.path.join(output_dir, "ctrate_train_paths.csv")
        },
        'valid': {
            'metadata': os.path.join(metadata_dir, "validation_metadata.csv"),
            'output': os.path.join(output_dir, "ctrate_valid_paths.csv")
        },
        'test': {
            'metadata': os.path.join(metadata_dir, "test_metadata.csv"),
            'output': os.path.join(output_dir, "ctrate_test_paths.csv")
        }
    }
    
    # Process each split
    total_found = 0
    total_missing = 0
    
    for split_name, config in splits.items():
        if not os.path.exists(config['metadata']):
            print(f"❌ Metadata file not found: {config['metadata']}")
            continue
            
        found, missing = create_split_csv(
            split_name,
            config['metadata'], 
            search_dirs,
            config['output']
        )
        total_found += found
        total_missing += missing
    
    print(f"\n=== SUMMARY ===")
    print(f"Total volumes found: {total_found}")
    print(f"Total volumes missing: {total_missing}")
    print("\nGenerated CSV files:")
    for split_name, config in splits.items():
        if os.path.exists(config['output']):
            print(f"  - {config['output']}")

if __name__ == "__main__":
    main()