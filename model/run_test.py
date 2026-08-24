#!/usr/bin/env python3
"""
Test script for evaluating trained CLIP-3D-CT models on CT-RATE test set.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import h5py
import json
from datetime import datetime

from eval import evaluate
from train import load_clip
import clip


def parse_args():
    parser = argparse.ArgumentParser(description='Test CLIP-3D-CT model on CT-RATE dataset')
    
    # Model paths
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model_name', type=str, default='ct-clip-test',
                       help='Name for saving results')
    
    # Model parameters
    parser.add_argument('--context_length', type=int, default=77)
    parser.add_argument('--dinov2_model_name', type=str, default='dinov2_vitb14')
    parser.add_argument('--dino_version', type=str, default='v2', choices=['v2', 'v3'],
                        help='Choose between DinoV2 and DinoV3')
    parser.add_argument('--freeze_dinov2', action='store_true')
    
    # Fusion method parameters
    parser.add_argument('--fusion_method', type=str, default='transformer', 
                       choices=['transformer', 'attentive'],
                       help='Slice fusion method: transformer (x_transformers) or attentive (VJEPA-style)')
    parser.add_argument('--fusion_depth', type=int, default=4,
                       help='Depth of the fusion module (number of layers)')
    
    # CT-RATE test paths
    parser.add_argument('--ctrate_test_ct_path', type=str, 
                       default='/path/to/data/ctrate_test.h5',
                       help='Path to CT-RATE test HDF5 file')
    parser.add_argument('--ctrate_test_label_path', type=str,
                       default='/path/to/ACT/model/data/ct_rate/test_predicted_labels.csv',
                       help='Path to CT-RATE test labels')
    
    
    # Test parameters
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of DataLoader workers')
    parser.add_argument('--save_dir', type=str, default='test_results/',
                       help='Directory to save test results')
    
    # Flags
    parser.add_argument('--test_ctrate', action='store_true', default=True,
                       help='Test on CT-RATE dataset')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save prediction probabilities')
    
    return parser.parse_args()


class CTTestDataset(torch.utils.data.Dataset):
    """Test dataset for CT volumes only."""
    def __init__(self, img_path, volume_names):
        self.img_dset = h5py.File(img_path, 'r')['ct_volumes']
        self.volume_names = volume_names
        self.num_volumes = len(volume_names)
        print(f"Test dataset: {self.img_dset.shape[0]} volumes in HDF5, {self.num_volumes} in labels CSV")
        
    def __len__(self):
        return self.num_volumes
    
    def __getitem__(self, idx):
        img = self.img_dset[idx]  # (D, H, W)
        img = np.expand_dims(img, axis=0)  # Add channel: (1, D, H, W)
        img = np.repeat(img, 3, axis=0)  # Repeat for RGB: (3, D, H, W)
        img = torch.from_numpy(img).float()
        return {'img': img, 'idx': idx, 'volume_name': self.volume_names[idx]}


def test_ctrate(model, config, device):
    """Test model on CT-RATE test set."""
    print("\n" + "="*60)
    print("Testing on CT-RATE Test Set")
    print("="*60)
    
    if not os.path.exists(config.ctrate_test_ct_path) or not os.path.exists(config.ctrate_test_label_path):
        print(f"CT-RATE test files not found. Skipping.")
        return None
    
    # Load test labels
    test_df = pd.read_csv(config.ctrate_test_label_path)
    
    # CT-RATE labels (18 pathologies)
    ctrate_labels = ['Medical material', 'Arterial wall calcification', 'Cardiomegaly', 
                     'Pericardial effusion', 'Coronary artery wall calcification', 'Hiatal hernia',
                     'Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule',
                     'Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion',
                     'Mosaic attenuation pattern', 'Peribronchial thickening', 'Consolidation',
                     'Bronchiectasis', 'Interlobular septal thickening']
    
    # Extract ground truth
    y_true = test_df[ctrate_labels].values
    volume_names = test_df['VolumeName'].values
    
    print(f"Found {len(y_true)} test samples with {len(ctrate_labels)} labels")
    
    # Create test dataset and loader
    test_dataset = CTTestDataset(config.ctrate_test_ct_path, volume_names)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Templates for zero-shot classification
    templates = [("{}", "no {}")]
    pos_template, neg_template = templates[0]
    
    # Encode text templates
    model.eval()
    with torch.no_grad():
        pos_texts = [pos_template.format(c) for c in ctrate_labels]
        neg_texts = [neg_template.format(c) for c in ctrate_labels]
        
        context_length = getattr(model, 'context_length', config.context_length)
        pos_tokens = clip.tokenize(pos_texts, context_length).to(device)
        neg_tokens = clip.tokenize(neg_texts, context_length).to(device)
        
        pos_features = model.encode_text(pos_tokens)
        neg_features = model.encode_text(neg_tokens)
        pos_features /= pos_features.norm(dim=-1, keepdim=True)
        neg_features /= neg_features.norm(dim=-1, keepdim=True)
    
    # Extract image features
    all_img_features = []
    all_volume_names = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extracting CT-RATE features"):
            imgs = batch['img'].to(device)
            feats = model.encode_image(imgs)
            feats /= feats.norm(dim=-1, keepdim=True)
            all_img_features.append(feats.cpu())
            all_volume_names.extend(batch['volume_name'])
    
    # Compute predictions
    img_features = torch.cat(all_img_features).to(device)
    logits_pos = img_features @ pos_features.T
    logits_neg = img_features @ neg_features.T
    probs = torch.exp(logits_pos) / (torch.exp(logits_pos) + torch.exp(logits_neg))
    y_pred = probs.cpu().numpy()
    
    # Evaluate
    results_df = evaluate(y_pred, y_true[:len(y_pred)], ctrate_labels)
    
    # Calculate overall metrics
    auc_cols = [col for col in results_df.columns if col.endswith('_auc')]
    mean_auc = results_df[auc_cols].mean().mean() if auc_cols else 0
    
    print(f"\nCT-RATE Overall Mean AUC: {mean_auc:.4f}")
    print("\nIndividual AUC scores:")
    for col in sorted(auc_cols):
        pathology = col.replace('_auc', '')
        auc_score = results_df[col].iloc[0]
        print(f"  {pathology}: {auc_score:.4f}")
    
    # Prepare results dictionary
    results = {
        'dataset': 'CT-RATE',
        'num_samples': len(y_pred),
        'num_labels': len(ctrate_labels),
        'mean_auc': mean_auc,
        'individual_aucs': {col.replace('_auc', ''): results_df[col].iloc[0] for col in auc_cols},
        'predictions': y_pred.tolist() if config.save_predictions else None,
        'volume_names': all_volume_names if config.save_predictions else None
    }
    
    return results, results_df




def main():
    config = parse_args()
    
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(config.save_dir, f"{config.model_name}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Results will be saved to: {results_dir}")
    
    # Load model
    print(f"\nLoading model from: {config.model_path}")
    model = load_clip(
        model_path=config.model_path,
        context_length=config.context_length,
        dinov2_model_name=config.dinov2_model_name,
        dino_version=config.dino_version,
        freeze_dinov2=config.freeze_dinov2,
        fusion_method=config.fusion_method,
        fusion_depth=config.fusion_depth
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    # Test on datasets
    all_results = {}
    
    if config.test_ctrate:
        ctrate_results, ctrate_df = test_ctrate(model, config, device)
        if ctrate_results:
            all_results['ctrate'] = ctrate_results
            # Save detailed results
            ctrate_df.to_csv(os.path.join(results_dir, 'ctrate_test_results.csv'), index=False)
            print(f"CT-RATE results saved to: {results_dir}/ctrate_test_results.csv")
    
    # Save summary results
    summary_path = os.path.join(results_dir, 'test_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary results saved to: {summary_path}")
    
    # Print final summary
    print("\n" + "="*60)
    print("TESTING COMPLETED")
    print("="*60)
    
    if 'ctrate' in all_results:
        print(f"CT-RATE Mean AUC: {all_results['ctrate']['mean_auc']:.4f}")
    
    print(f"\nAll results saved to: {results_dir}")


if __name__ == "__main__":
    main()