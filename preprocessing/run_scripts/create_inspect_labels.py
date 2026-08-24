#!/usr/bin/env python3
"""
Create label files for Inspect dataset compatible with CT-RATE format.
Focuses on pulmonary embolism detection and available medical findings.
"""

import pandas as pd
import os

def create_inspect_labels():
    # Base paths
    base_path = "/path/to/data/Inspect_v2.0/full"
    output_dir = "/path/to/ACT/model/data/inspect"
    
    # Read the TSV files
    splits_df = pd.read_csv(os.path.join(base_path, "splits_20250611.tsv"), sep='\t')
    mapping_df = pd.read_csv(os.path.join(base_path, "study_mapping_20250611.tsv"), sep='\t')
    labels_df = pd.read_csv(os.path.join(base_path, "labels_20250611.tsv"), sep='\t')
    
    # Merge to get complete dataset
    merged_df = splits_df.merge(mapping_df[['impression_id', 'image_id']], on='impression_id', how='left')
    merged_df = merged_df.merge(labels_df, on='impression_id', how='left')
    
    # CT-RATE compatible columns (focus on PE and available findings)
    # PE-related: pe_positive (main target)
    # Available findings from labels: Atelectasis, Cardiomegaly, Consolidation, Pleural_Effusion
    # Missing findings filled with 0 (conservative approach)
    ctrate_columns = [
        'VolumeName',
        'Medical material', 'Arterial wall calcification', 'Cardiomegaly', 'Pericardial effusion',
        'Coronary artery wall calcification', 'Hiatal hernia', 'Lymphadenopathy', 'Emphysema',
        'Atelectasis', 'Lung nodule', 'Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion',
        'Mosaic attenuation pattern', 'Peribronchial thickening', 'Consolidation', 'Bronchiectasis',
        'Interlobular septal thickening'
    ]
    
    # Create labels for each split
    for split in ['train', 'valid', 'test']:
        split_df = merged_df[merged_df['split'] == split].copy()
        
        # Create VolumeName (image_id + .nii.gz)
        split_df['VolumeName'] = split_df['image_id'] + '.nii.gz'
        
        # Create label DataFrame with CT-RATE structure
        labels_output_df = pd.DataFrame()
        labels_output_df['VolumeName'] = split_df['VolumeName']
        
        # Map available findings from Inspect to CT-RATE format
        # Focus on PE as the main finding, set others to 0 for compatibility
        labels_output_df['Medical material'] = 0  # Not available
        labels_output_df['Arterial wall calcification'] = 0  # Not available  
        labels_output_df['Cardiomegaly'] = 0  # Not available as binary label
        labels_output_df['Pericardial effusion'] = 0  # Not available
        labels_output_df['Coronary artery wall calcification'] = 0  # Not available
        labels_output_df['Hiatal hernia'] = 0  # Not available
        labels_output_df['Lymphadenopathy'] = 0  # Not available  
        labels_output_df['Emphysema'] = 0  # Not available
        labels_output_df['Atelectasis'] = 0  # Not available as binary label
        labels_output_df['Lung nodule'] = 0  # Not available
        labels_output_df['Lung opacity'] = 0  # Not available
        labels_output_df['Pulmonary fibrotic sequela'] = 0  # Not available
        labels_output_df['Pleural effusion'] = 0  # Not available as binary label
        labels_output_df['Mosaic attenuation pattern'] = 0  # Not available
        labels_output_df['Peribronchial thickening'] = 0  # Not available
        labels_output_df['Consolidation'] = 0  # Not available as binary label
        labels_output_df['Bronchiectasis'] = 0  # Not available
        labels_output_df['Interlobular septal thickening'] = 0  # Not available
        
        # Handle NaN values by setting to 0
        labels_output_df = labels_output_df.fillna(0)
        
        # Save predicted labels file  
        labels_output_path = os.path.join(output_dir, f"{split}_predicted_labels.csv")
        labels_output_df.to_csv(labels_output_path, index=False)
        print(f"Saved {len(labels_output_df)} {split} labels to: {labels_output_path}")
        
        # Also create a PE-specific labels file for the main task
        pe_labels_df = pd.DataFrame()
        pe_labels_df['VolumeName'] = split_df['VolumeName']
        pe_labels_df['pe_positive'] = split_df['pe_positive'].fillna(0).astype(int)
        pe_labels_df['pe_acute'] = split_df['pe_acute'].fillna(0).astype(int)
        pe_labels_df['pe_subsegmentalonly'] = split_df['pe_subsegmentalonly'].fillna(0).astype(int)
        
        pe_output_path = os.path.join(output_dir, f"{split}_pe_labels.csv")
        pe_labels_df.to_csv(pe_output_path, index=False)
        print(f"Saved {len(pe_labels_df)} {split} PE labels to: {pe_output_path}")
        
        # Print statistics
        print(f"{split.upper()} split statistics:")
        print(f"  Total samples: {len(split_df)}")
        print(f"  PE positive: {pe_labels_df['pe_positive'].sum()} ({pe_labels_df['pe_positive'].mean():.1%})")
        print(f"  PE acute: {pe_labels_df['pe_acute'].sum()} ({pe_labels_df['pe_acute'].mean():.1%})")
        print(f"  Other findings: All set to 0 (focus on PE detection)")
        print()

if __name__ == "__main__":
    create_inspect_labels()