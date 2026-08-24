#!/usr/bin/env python3
"""
Generate CSV files with CT paths for each split (train/val/test) for the Merlin dataset.
Creates files similar to ctrate_*_paths.csv and inspect_*_paths.csv.
"""

import pandas as pd
import os
from pathlib import Path

def generate_merlin_split_csvs():
    # Base paths
    base_path = "/path/to/data/Merlin/merlinabdominalctdataset"
    ct_path = os.path.join(base_path, "merlin_data")
    output_dir = "/path/to/ACT/preprocessing/run_scripts"
    
    # Read the Excel file (converted to understanding from exploration)
    excel_path = os.path.join(base_path, "reports_final.xlsx")
    df = pd.read_excel(excel_path)
    
    print(f"Total samples in Excel: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Split distribution:")
    print(df['Split'].value_counts())
    
    # Check how many CT files we actually have
    ct_files = set()
    for file in os.listdir(ct_path):
        if file.endswith('.nii.gz'):
            # Remove .nii.gz extension to get study_id
            study_id = file.replace('.nii.gz', '')
            ct_files.add(study_id)
    
    print(f"\nActual CT files found: {len(ct_files)}")
    
    # Check matching between Excel and actual files
    excel_ids = set(df['study id'].tolist())
    missing_in_excel = ct_files - excel_ids
    missing_files = excel_ids - ct_files
    
    print(f"CT files missing from Excel: {len(missing_in_excel)}")
    print(f"Excel entries missing CT files: {len(missing_files)}")
    if missing_files:
        print(f"First 10 missing files: {list(missing_files)[:10]}")
    
    # Create CSV files for each split, only including files that exist
    split_mapping = {'train': 'train', 'val': 'valid', 'test': 'test'}
    
    for excel_split, our_split in split_mapping.items():
        split_df = df[df['Split'] == excel_split].copy()
        
        # Filter to only include files that actually exist
        split_df = split_df[split_df['study id'].isin(ct_files)]
        
        # Create full paths to CT files
        split_df['Path'] = split_df['study id'].apply(lambda x: os.path.join(ct_path, f"{x}.nii.gz"))
        
        # Verify all files exist (should be true after filtering)
        missing_files_in_split = []
        for _, row in split_df.iterrows():
            if not os.path.exists(row['Path']):
                missing_files_in_split.append(row['Path'])
        
        if missing_files_in_split:
            print(f"Warning: {len(missing_files_in_split)} missing files in {our_split} split")
            split_df = split_df[split_df['Path'].apply(os.path.exists)]
        
        # Save paths CSV (compatible with existing preprocessing pipeline)
        paths_output = os.path.join(output_dir, f"merlin_{our_split}_paths.csv")
        split_df[['Path']].to_csv(paths_output, index=False)
        print(f"Saved {len(split_df)} {our_split} paths to: {paths_output}")
        
        # Save reports CSV for index-wise pairing
        reports_output = os.path.join(output_dir, f"merlin_{our_split}_reports.csv")
        # Rename columns to match expected format
        reports_df = split_df[['study id', 'Findings']].copy()
        reports_df = reports_df.rename(columns={'study id': 'VolumeName', 'Findings': 'impressions'})
        # Add .nii.gz extension to VolumeName for consistency
        reports_df['VolumeName'] = reports_df['VolumeName'] + '.nii.gz'
        reports_df.to_csv(reports_output, index=False)
        print(f"Saved {len(reports_df)} {our_split} reports to: {reports_output}")
        
        # Save metadata CSV with all information
        metadata_output = os.path.join(output_dir, f"merlin_{our_split}_metadata.csv")
        metadata_df = split_df.copy()
        metadata_df['VolumeName'] = metadata_df['study id'] + '.nii.gz'
        metadata_df.to_csv(metadata_output, index=False)
        print(f"Saved {len(metadata_df)} {our_split} metadata entries to: {metadata_output}")
    
    # Summary statistics
    print(f"\n=== MERLIN DATASET SUMMARY ===")
    total_usable = 0
    for excel_split, our_split in split_mapping.items():
        split_count = len(df[(df['Split'] == excel_split) & (df['study id'].isin(ct_files))])
        total_usable += split_count
        print(f"{our_split.upper()}: {split_count} samples")
    
    print(f"TOTAL USABLE: {total_usable} samples")
    print(f"Dataset type: Abdominal CT with radiology findings")

if __name__ == "__main__":
    generate_merlin_split_csvs()