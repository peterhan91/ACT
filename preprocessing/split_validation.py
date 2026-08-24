#!/usr/bin/env python3
"""
Patient-wise splitting of CT-RATE validation set into new validation and test sets.
Target: ~500 CTs for validation, rest for test.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import argparse
from collections import defaultdict

def extract_patient_id(volume_name):
    """Extract patient ID from volume name like 'valid_123_a_1.nii.gz' -> 123"""
    parts = volume_name.split('_')
    if len(parts) >= 2 and parts[0] == 'valid':
        try:
            return int(parts[1])
        except ValueError:
            pass
    return None

def analyze_patient_distribution(df):
    """Analyze the distribution of volumes per patient."""
    patient_volumes = defaultdict(list)
    
    for _, row in df.iterrows():
        patient_id = extract_patient_id(row['VolumeName'])
        if patient_id is not None:
            patient_volumes[patient_id].append(row['VolumeName'])
    
    print(f"Total unique patients: {len(patient_volumes)}")
    print(f"Total volumes: {len(df)}")
    
    # Analyze distribution
    volumes_per_patient = [len(volumes) for volumes in patient_volumes.values()]
    print(f"Volumes per patient - Min: {min(volumes_per_patient)}, Max: {max(volumes_per_patient)}, Mean: {np.mean(volumes_per_patient):.2f}")
    
    return patient_volumes

def select_validation_patients(patient_volumes, target_validation_size=500):
    """
    Select patients for validation set to get approximately target_validation_size volumes.
    Strategy: Prefer patients with fewer volumes to balance the split.
    """
    # Sort patients by number of volumes (ascending)
    patients_sorted = sorted(patient_volumes.items(), key=lambda x: len(x[1]))
    
    validation_patients = []
    validation_volume_count = 0
    
    print(f"Selecting patients for validation set (target: ~{target_validation_size} volumes)...")
    
    for patient_id, volumes in patients_sorted:
        if validation_volume_count + len(volumes) <= target_validation_size + 50:  # Allow some buffer
            validation_patients.append(patient_id)
            validation_volume_count += len(volumes)
            print(f"  Added patient {patient_id} ({len(volumes)} volumes) - Total: {validation_volume_count}")
        else:
            # Check if we're still far from target
            if validation_volume_count < target_validation_size - 100:
                validation_patients.append(patient_id)
                validation_volume_count += len(volumes)
                print(f"  Added patient {patient_id} ({len(volumes)} volumes) - Total: {validation_volume_count}")
    
    print(f"Selected {len(validation_patients)} patients with {validation_volume_count} total volumes for validation")
    
    return set(validation_patients)

def split_dataframes(metadata_df, reports_df, labels_df, validation_patients):
    """Split the dataframes based on validation patient IDs."""
    
    def is_validation_volume(volume_name):
        patient_id = extract_patient_id(volume_name)
        return patient_id in validation_patients
    
    # Split metadata
    val_metadata = metadata_df[metadata_df['VolumeName'].apply(is_validation_volume)].copy()
    test_metadata = metadata_df[~metadata_df['VolumeName'].apply(is_validation_volume)].copy()
    
    # Split reports
    val_reports = reports_df[reports_df['VolumeName'].apply(is_validation_volume)].copy()
    test_reports = reports_df[~reports_df['VolumeName'].apply(is_validation_volume)].copy()
    
    # Split labels
    val_labels = labels_df[labels_df['VolumeName'].apply(is_validation_volume)].copy()
    test_labels = labels_df[~labels_df['VolumeName'].apply(is_validation_volume)].copy()
    
    print(f"Validation set: {len(val_metadata)} volumes")
    print(f"Test set: {len(test_metadata)} volumes")
    
    return (val_metadata, test_metadata), (val_reports, test_reports), (val_labels, test_labels)

def main():
    parser = argparse.ArgumentParser(description='Split CT-RATE validation set into validation and test sets')
    parser.add_argument('--data_dir', type=str, default='data/ct_rate', 
                        help='Directory containing CT-RATE CSV files')
    parser.add_argument('--target_val_size', type=int, default=500,
                        help='Target number of volumes for validation set')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    data_dir = Path(args.data_dir)
    
    # Load the current validation files
    print("Loading validation data...")
    metadata_df = pd.read_csv(data_dir / 'validation_metadata.csv')
    reports_df = pd.read_csv(data_dir / 'validation_reports.csv')
    labels_df = pd.read_csv(data_dir / 'valid_predicted_labels.csv')
    
    print(f"Loaded {len(metadata_df)} volumes from validation set")
    
    # Analyze patient distribution
    patient_volumes = analyze_patient_distribution(metadata_df)
    
    # Select validation patients
    validation_patients = select_validation_patients(patient_volumes, args.target_val_size)
    
    # Split the dataframes
    (val_metadata, test_metadata), (val_reports, test_reports), (val_labels, test_labels) = split_dataframes(
        metadata_df, reports_df, labels_df, validation_patients
    )
    
    # Save the new splits
    print("\nSaving new validation and test sets...")
    
    # New validation files
    val_metadata.to_csv(data_dir / 'new_validation_metadata.csv', index=False)
    val_reports.to_csv(data_dir / 'new_validation_reports.csv', index=False)
    val_labels.to_csv(data_dir / 'new_valid_predicted_labels.csv', index=False)
    
    # New test files
    test_metadata.to_csv(data_dir / 'test_metadata.csv', index=False)
    test_reports.to_csv(data_dir / 'test_reports.csv', index=False)
    test_labels.to_csv(data_dir / 'test_predicted_labels.csv', index=False)
    
    print(f"✓ Saved new validation set: {len(val_metadata)} volumes")
    print(f"✓ Saved new test set: {len(test_metadata)} volumes")
    
    # Create a summary file
    summary = {
        'total_original_volumes': len(metadata_df),
        'validation_volumes': len(val_metadata),
        'test_volumes': len(test_metadata),
        'validation_patients': len(validation_patients),
        'test_patients': len(patient_volumes) - len(validation_patients),
        'target_validation_size': args.target_val_size,
        'random_seed': args.seed,
        'validation_patient_ids': sorted(list(validation_patients))
    }
    
    import json
    with open(data_dir / 'split_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Saved split summary to {data_dir / 'split_summary.json'}")
    print("\nSplit Summary:")
    print(f"  Original validation set: {summary['total_original_volumes']} volumes")
    print(f"  New validation set: {summary['validation_volumes']} volumes ({summary['validation_patients']} patients)")
    print(f"  New test set: {summary['test_volumes']} volumes ({summary['test_patients']} patients)")

if __name__ == "__main__":
    main()