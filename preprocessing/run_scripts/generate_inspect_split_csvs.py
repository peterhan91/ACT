#!/usr/bin/env python3
"""
Generate CSV files with CT paths for each split (train/valid/test) for the Inspect dataset.
Creates files similar to ctrate_*_paths.csv but for Inspect dataset.
"""

import pandas as pd
import os
from pathlib import Path

def generate_inspect_split_csvs():
    # Base paths
    base_path = "/path/to/data/Inspect_v2.0/full"
    ctpa_path = os.path.join(base_path, "CTPA")
    output_dir = "/path/to/ACT/preprocessing/run_scripts"
    
    # Read the TSV files
    splits_df = pd.read_csv(os.path.join(base_path, "splits_20250611.tsv"), sep='\t')
    mapping_df = pd.read_csv(os.path.join(base_path, "study_mapping_20250611.tsv"), sep='\t')
    impressions_df = pd.read_csv(os.path.join(base_path, "impressions_20250611.tsv"), sep='\t')
    
    # Merge splits with mapping to get image_ids
    merged_df = splits_df.merge(mapping_df[['impression_id', 'image_id']], on='impression_id', how='left')
    
    # Also merge with impressions for the reports CSV
    merged_with_text_df = merged_df.merge(impressions_df, on='impression_id', how='left')
    
    print(f"Total samples: {len(merged_df)}")
    print(f"Split distribution:")
    print(merged_df['split'].value_counts())
    
    # Create CSV files for each split
    for split in ['train', 'valid', 'test']:
        split_df = merged_df[merged_df['split'] == split].copy()
        
        # Create full paths to CT files
        split_df['Path'] = split_df['image_id'].apply(lambda x: os.path.join(ctpa_path, f"{x}.nii.gz"))
        
        # Check if files exist
        missing_files = []
        for _, row in split_df.iterrows():
            if not os.path.exists(row['Path']):
                missing_files.append(row['Path'])
        
        if missing_files:
            print(f"Warning: {len(missing_files)} missing files in {split} split")
            print(f"First 5 missing files: {missing_files[:5]}")
            # Remove missing files
            split_df = split_df[split_df['Path'].apply(os.path.exists)]
        
        # Save paths CSV (compatible with existing preprocessing pipeline)
        paths_output = os.path.join(output_dir, f"inspect_{split}_paths.csv")
        split_df[['Path']].to_csv(paths_output, index=False)
        print(f"Saved {len(split_df)} {split} paths to: {paths_output}")
        
        # Save reports CSV for index-wise pairing
        split_text_df = merged_with_text_df[merged_with_text_df['split'] == split].copy()
        split_text_df['Path'] = split_text_df['image_id'].apply(lambda x: os.path.join(ctpa_path, f"{x}.nii.gz"))
        # Filter out any missing files from text data too
        split_text_df = split_text_df[split_text_df['Path'].apply(os.path.exists)]
        
        reports_output = os.path.join(output_dir, f"inspect_{split}_reports.csv")
        split_text_df[['impression_id', 'person_id', 'image_id', 'impressions']].to_csv(reports_output, index=False)
        print(f"Saved {len(split_text_df)} {split} reports to: {reports_output}")
        
        # Save metadata CSV with all information (reuse the split_df that already has the Path column)
        metadata_output = os.path.join(output_dir, f"inspect_{split}_metadata.csv")
        # Use the full mapping data for this split
        full_metadata_df = split_df.merge(
            mapping_df.drop('image_id', axis=1), on='impression_id', how='left'
        )
        full_metadata_df.to_csv(metadata_output, index=False)
        print(f"Saved {len(full_metadata_df)} {split} metadata entries to: {metadata_output}")

if __name__ == "__main__":
    generate_inspect_split_csvs()