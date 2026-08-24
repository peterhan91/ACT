#!/usr/bin/env python3
"""
Script to check validation H5 files for corrupted or problematic CT volumes.
"""
import h5py
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def check_h5_file(h5_path, dataset_name="ct_volumes", sample_size=100):
    """Check H5 file for data integrity issues with memory-efficient sampling."""
    print(f"\n=== Checking {h5_path} ===")
    
    if not os.path.exists(h5_path):
        print(f"ERROR: File does not exist: {h5_path}")
        return False
    
    try:
        with h5py.File(h5_path, 'r') as f:
            print(f"File opened successfully")
            print(f"Available datasets: {list(f.keys())}")
            
            if dataset_name not in f:
                print(f"ERROR: Dataset '{dataset_name}' not found in file")
                return False
            
            dataset = f[dataset_name]
            print(f"Dataset shape: {dataset.shape}")
            print(f"Dataset dtype: {dataset.dtype}")
            
            num_volumes = dataset.shape[0]
            print(f"Number of volumes: {num_volumes}")
            
            # Memory-efficient sampling: check subset of volumes + specific known issues
            if num_volumes <= sample_size:
                indices_to_check = list(range(num_volumes))
                print(f"Checking all {num_volumes} volumes")
            else:
                # Sample evenly distributed indices
                step = max(1, num_volumes // sample_size)
                indices_to_check = list(range(0, num_volumes, step))
                # Add known problematic indices if they exist
                known_issues = [853]  # Volume 853 known to be corrupted in inspect_valid
                for idx in known_issues:
                    if idx < num_volumes and idx not in indices_to_check:
                        indices_to_check.append(idx)
                indices_to_check.sort()
                print(f"Checking {len(indices_to_check)} sampled volumes (every {step}th volume)")
            
            # Check selected volumes
            problematic_volumes = []
            for i in tqdm(indices_to_check, desc="Checking volumes"):
                try:
                    # Load only volume metadata first (shape check)
                    volume_shape = dataset[i].shape
                    if len(volume_shape) != 3:
                        problematic_volumes.append((i, f"Wrong shape: {volume_shape}"))
                        continue
                    
                    # Load actual volume data for content checks
                    volume = dataset[i]
                    
                    # Quick checks using numpy array methods (more memory efficient)
                    if np.all(volume == 0):
                        problematic_volumes.append((i, "All zeros"))
                        continue
                    
                    # Check for NaN/inf using numpy's fast methods
                    if np.any(np.isnan(volume)):
                        problematic_volumes.append((i, "Contains NaN values"))
                        continue
                    
                    if np.any(np.isinf(volume)):
                        problematic_volumes.append((i, "Contains infinite values"))
                        continue
                    
                    # Check for extreme values (compute min/max once)
                    vol_min, vol_max = np.min(volume), np.max(volume)
                    if vol_min < -2000 or vol_max > 4000:  # Typical HU range
                        problematic_volumes.append((i, f"Extreme values: min={vol_min}, max={vol_max}"))
                        continue
                    
                    # Clear volume from memory immediately
                    del volume
                    
                except Exception as e:
                    problematic_volumes.append((i, f"Loading error: {str(e)}"))
            
            # Report results
            print(f"\nResults:")
            print(f"Total volumes in file: {num_volumes}")
            print(f"Volumes checked: {len(indices_to_check)}")
            print(f"Problematic volumes found: {len(problematic_volumes)}")
            
            if problematic_volumes:
                print("\nProblematic volumes:")
                for idx, issue in problematic_volumes[:10]:  # Show first 10
                    print(f"  Volume {idx}: {issue}")
                if len(problematic_volumes) > 10:
                    print(f"  ... and {len(problematic_volumes) - 10} more")
                
                return False
            else:
                print("All checked volumes passed integrity checks!")
                return True
                
    except Exception as e:
        print(f"ERROR opening file: {str(e)}")
        return False

def check_csv_file(csv_path):
    """Check CSV file for issues."""
    print(f"\n=== Checking {csv_path} ===")
    
    if not os.path.exists(csv_path):
        print(f"ERROR: File does not exist: {csv_path}")
        return False
    
    try:
        df = pd.read_csv(csv_path)
        print(f"CSV shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check for missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            print(f"Columns with missing values: {missing_cols}")
            for col in missing_cols:
                missing_count = df[col].isnull().sum()
                print(f"  {col}: {missing_count} missing values")
        
        # Check for duplicate volume names
        if 'VolumeName' in df.columns:
            duplicates = df['VolumeName'].duplicated().sum()
            if duplicates > 0:
                print(f"WARNING: {duplicates} duplicate volume names")
        
        return True
        
    except Exception as e:
        print(f"ERROR reading CSV: {str(e)}")
        return False

def main():
    files_to_check = [
        # Validation files (use sampling)
        ("/path/to/data/ctrate_valid.h5", "/path/to/ACT/model/data/ct_rate/valid_predicted_labels.csv", "CT-RATE Validation", 100),
        ("/path/to/data/inspect_valid.h5", "/path/to/ACT/model/data/inspect/valid_pe_labels.csv", "INSPECT Validation", 100),

        # Training files (check all volumes)
        ("/path/to/data/inspect_train.h5", "/path/to/ACT/model/data/inspect/train_reports.csv", "INSPECT Training", None),
        ("/path/to/data/merlin_train.h5", "/path/to/ACT/model/data/merlin/train_reports.csv", "MERLIN Training", None),
    ]
    
    results = []
    
    for h5_path, csv_path, dataset_name, sample_size in files_to_check:
        print(f"\nChecking {dataset_name} data...")
        if sample_size is None:
            print("NOTE: Checking ALL volumes in training dataset (this may take several minutes)")
            h5_ok = check_h5_file(h5_path, sample_size=float('inf'))  # Check all volumes
        else:
            h5_ok = check_h5_file(h5_path, sample_size=sample_size)
        csv_ok = check_csv_file(csv_path)
        results.append((dataset_name, h5_ok, csv_ok))
    
    # Summary
    print("\n" + "="*70)
    print("DATA INTEGRITY CHECK SUMMARY")
    print("="*70)
    
    all_ok = True
    for dataset_name, h5_ok, csv_ok in results:
        h5_status = '✓ OK' if h5_ok else '✗ ISSUES FOUND'
        csv_status = '✓ OK' if csv_ok else '✗ ISSUES FOUND'
        print(f"{dataset_name:<20} H5: {h5_status:<15} CSV: {csv_status}")
        if not (h5_ok and csv_ok):
            all_ok = False
    
    if all_ok:
        print("\n✓ All data files passed integrity checks!")
    else:
        print("\n✗ Some data files have issues that may cause training problems.")
        print("Consider using the corrupted volume detection in CTValidationDataset for problematic files.")

if __name__ == "__main__":
    main()