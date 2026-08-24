import os
import numpy as np
import pandas as pd
import h5py
from einops import rearrange

import torch
from torch.utils import data
from torch import nn
import torch.optim as optim
from x_transformers import Encoder

from model import CLIP
from simple_tokenizer import SimpleTokenizer
from attentive_pooler import AttentivePooler
# SigLIP loss is now in loss module


class CTDataset(data.Dataset):
    """Dataset for 3D CT volumes with text reports.
    
    Input params:
        img_path: Path to the HDF5 file containing CT volumes.
        txt_path: Path to the CSV file containing text reports.
        column: Column name in CSV containing the text reports (default='report').
        size: Optional size limit for dataset.
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(self, img_path, txt_path, column='report', size=None, transform=None):
        super().__init__()
        if size != None: 
            self.img_dset = h5py.File(img_path, 'r')['ct_volumes'][:size]
            self.txt_dset = pd.read_csv(txt_path)[column][:size]
        else: 
            self.img_dset = h5py.File(img_path, 'r')['ct_volumes']
            self.txt_dset = pd.read_csv(txt_path)[column]
        self.transform = transform
            
    def __len__(self):
        return len(self.txt_dset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img = self.img_dset[idx]  # np array, (D, H, W) - preprocessed HU values
        img = np.expand_dims(img, axis=0)  # Add channel dimension: (1, D, H, W)
        img = np.repeat(img, 3, axis=0)  # Repeat for RGB: (3, D, H, W)
        txt = self.txt_dset[idx]  # python str
        if pd.isna(txt) or txt == "":  # capture the case of empty sections
            txt = " "

        img = torch.from_numpy(img).float()  # torch, (3, D, H, W) - ensure float32
        if self.transform:
            img = self.transform(img)
        sample = {'img': img, 'txt': txt}
        
        return sample


class MultiCTDataset(data.Dataset):
    """Dataset for multiple 3D CT volumes with text reports."""
    def __init__(self, img_paths, txt_paths, columns='report', transform=None):
        super().__init__()
        assert len(img_paths) == len(txt_paths), "Number of image and text paths must match"
        
        if isinstance(columns, str):
            columns = [columns] * len(img_paths)
        elif isinstance(columns, list):
            assert len(columns) == len(img_paths), f"Number of columns ({len(columns)}) must match number of datasets ({len(img_paths)})"
        
        self.datasets = []
        self.cumulative_lengths = [0]
        self.transform = transform
        
        for img_path, txt_path, column in zip(img_paths, txt_paths, columns):
            img_dset = h5py.File(img_path, 'r')['ct_volumes']
            txt_dset = pd.read_csv(txt_path)[column]
            
            assert len(img_dset) == len(txt_dset), f"Mismatch in {img_path} and {txt_path}: {len(img_dset)} vs {len(txt_dset)}"
            
            print(f"Loading dataset {img_path}...")
            valid_indices = list(range(len(img_dset)))
            
            self.datasets.append((img_dset, txt_dset, valid_indices))
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(txt_dset))
        
        print(f"Loaded {len(self.datasets)} datasets with total {self.cumulative_lengths[-1]} samples")
        for i, (path, length) in enumerate(zip(img_paths, [self.cumulative_lengths[i+1] - self.cumulative_lengths[i] for i in range(len(self.datasets))])):
            print(f"  Dataset {i+1}: {length} samples from {path}")
    
    def __len__(self):
        return self.cumulative_lengths[-1]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Bounds check
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)-1}]")
        
        # Find which dataset this index belongs to
        dataset_idx = len(self.datasets) - 1  # Default to last dataset
        local_idx = idx
        for i in range(len(self.cumulative_lengths) - 1):
            if idx < self.cumulative_lengths[i + 1]:
                dataset_idx = i
                local_idx = idx - self.cumulative_lengths[i]
                break
        
        img_dset, txt_dset, valid_indices = self.datasets[dataset_idx]
        # Map dataset index to actual H5 index
        actual_idx = valid_indices[local_idx]
        img = img_dset[actual_idx]  # (D, H, W)
        txt = txt_dset.iloc[local_idx]
        
        # Process image
        img = np.expand_dims(img, axis=0)  # (1, D, H, W)
        img = np.repeat(img, 3, axis=0)  # (3, D, H, W)
        
        if pd.isna(txt) or txt == "":
            txt = " "
        
        img = torch.from_numpy(img).float()
        if self.transform:
            img = self.transform(img)
        
        return {'img': img, 'txt': txt}


def load_data(ct_filepath, txt_filepath, batch_size=4, column='report', verbose=False, num_workers=2, local_rank=0, rank=0, use_ddp=False, world_size=1): 
    if torch.cuda.is_available():  
        dev = f"cuda:{local_rank}" 
        cuda_available = True
        print(f'Using CUDA device {local_rank}.')
    else:  
        dev = "cpu"  
        cuda_available = False
        print('Using cpu.')
    
    device = torch.device(dev)
    
    if cuda_available: 
        torch.cuda.set_device(device)

    # For 3D CT volumes, no transforms needed - data is already preprocessed
    print("Loading 3D CT dataset - no transforms applied (data already preprocessed).")
    
    # Check if single files or multiple files
    if isinstance(ct_filepath, list) and isinstance(txt_filepath, list):
        if len(ct_filepath) == 1:
            # Single file passed as list
            torch_dset = CTDataset(img_path=ct_filepath[0],
                                  txt_path=txt_filepath[0], 
                                  column=column[0] if isinstance(column, list) else column, 
                                  transform=None)
        else:
            # Multiple files
            print(f"Loading multiple datasets: {len(ct_filepath)} H5 files")
            torch_dset = MultiCTDataset(img_paths=ct_filepath,
                                      txt_paths=txt_filepath, 
                                      columns=column, 
                                      transform=None)
    else:
        # Backward compatibility: single files as strings
        torch_dset = CTDataset(img_path=ct_filepath,
                              txt_path=txt_filepath, 
                              column=column if isinstance(column, str) else column[0], 
                              transform=None)
    
    if verbose: 
        for i in range(len(torch_dset)):
            sample = torch_dset[i]
            print(i, sample['img'].size(), sample['txt'])  # (3, D, H, W)
            if i == 3:
                break
    
    # Create sampler for DDP if needed
    if use_ddp:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(torch_dset, num_replicas=world_size, rank=rank, shuffle=True)
        loader_params = {
            'batch_size': batch_size, 
            'sampler': sampler, 
            'num_workers': num_workers,
            'pin_memory': True,
            'drop_last': True,  # Important for DDP to avoid uneven last batch
            'persistent_workers': True if num_workers > 0 else False
        }
    else:
        sampler = None
        loader_params = {
            'batch_size': batch_size, 
            'shuffle': True, 
            'num_workers': num_workers,
            'pin_memory': True,
            'persistent_workers': True if num_workers > 0 else False
        }
    
    data_loader = data.DataLoader(torch_dset, **loader_params)
    return data_loader, device, sampler
    

def load_clip(model_path=None, context_length=77, 
              dinov2_model_name="dinov2_vitb14", dino_version="v2", freeze_dinov2=False, local_rank=0,
              fusion_method="transformer", fusion_depth=4):
    '''
    FUNCTION: load_clip
    -------------------------------
    This function loads in a CLIP model with 3D DinoV2 vision encoder
    for CT volume processing.
    
    args: 
        * model_path (optional) - path to model weights that the model
        will be initialized with 
        * context_length (optional) - length of the maximum number of 
        tokens that can be inputted into the CLIP model
        * dinov2_model_name (optional) - DinoV2 model variant to use
        * dino_version (optional) - "v2" or "v3" to choose between DinoV2 and DinoV3
        * freeze_dinov2 (optional) - if True, freeze DinoV2 backbone
        * fusion_method (optional) - "transformer" or "attentive" for slice fusion
        * fusion_depth (optional) - depth of fusion module (default: 4)
    '''

    params = {
        'embed_dim':768,
        'image_resolution': 224,  # Updated for CT processing
        'vision_layers': 12,
        'vision_width': 768,
        'vision_patch_size': 16,
        'context_length': context_length,
        'vocab_size': 49408,
        'transformer_width': 512,
        'transformer_heads': 8,
        'transformer_layers': 12
    }
    
    # set device 
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # Load Dino backbone based on version (moved before model creation to get dimensions)
    if dino_version == "v3":
        # Load DinoV3 from local implementation
        if "vitl" in dinov2_model_name.lower():
            # Load DinoV3 VIT-L model
            dinov2_backbone = torch.hub.load('/path/to/dinov3', 'dinov3_vitl16',
                                            source='local',
                                            weights='/path/to/dinov3/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')
            print("Loading DinoV3 vitl16 model from local path")
        else:
            # Load DinoV3 VIT-B model (default)
            dinov2_backbone = torch.hub.load('/path/to/dinov3', 'dinov3_vitb16',
                                            source='local',
                                            weights='/path/to/dinov3/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')
            print("Loading DinoV3 vitb16 model from local path")
    else:
        # Load DinoV2 backbone from official Facebook Research implementation
        dinov2_backbone = torch.hub.load('facebookresearch/dinov2', dinov2_model_name+'_reg', pretrained=True)
    
    dinov2_backbone = dinov2_backbone.to(device)  # Move to correct device
    
    # Get feature dimension using a dummy forward pass
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).to(device)  # Move dummy input to device
        features = dinov2_backbone(dummy_input)  # forward() returns CLS token directly
        backbone_dim = features.shape[-1]
    
    # Update embed_dim to match backbone dimension (fixes VIT-L dimension mismatch)
    params['embed_dim'] = backbone_dim
    print(f"Setting embed_dim to {backbone_dim} to match {dino_version.upper()} backbone dimension")
    
    # Create CLIP model with updated dimensions
    model = CLIP(**params)
    
    # Fix initialization for larger dimensions
    if backbone_dim > 768:
        # Re-initialize text projection with better scaling for 1024 dims
        # Use Xavier/Glorot initialization which scales with dimensions
        std = np.sqrt(2.0 / (model.text_projection.shape[0] + model.text_projection.shape[1]))
        nn.init.normal_(model.text_projection, std=std)
        print(f"Re-initialized text projection for {backbone_dim}D with std={std:.4f}")
        
        # Use original logit scale - the adjustment was causing issues
        # Keep the standard CLIP initialization
        model.logit_scale.data = torch.ones([]) * np.log(1 / 0.07)
        print(f"Using standard logit_scale: exp({model.logit_scale.item():.3f}) = {model.logit_scale.exp().item():.3f}")
    
    # 3D version with slice fusion
    class DinoV2Visual3D(nn.Module):
        def __init__(self, backbone, backbone_dim, output_dim, fusion_method="transformer", fusion_depth=4):
            super().__init__()
            self.backbone = backbone
            self.fusion_method = fusion_method
            self.use_gradient_checkpointing = False
            
            if fusion_method == "transformer":
                # Slice-fusion Transformer. heads=12 / ff_mult=2 is the released
                # clip_3d_ctrate_merlin_v1 configuration (12 heads, feed-forward
                # multiplier 2 in the paper); changing these breaks checkpoint loading.
                self.slice_fusion = Encoder(
                    dim=backbone_dim,
                    heads=12,
                    ff_mult=2,
                    attn_dropout=0.0,
                    pre_norm=True,
                    depth=fusion_depth,
                    attn_flash=True,
                    ff_no_bias=True, 
                    rotary_pos_emb=True,
                )
                self.cls_token = nn.Parameter(torch.randn(1, 1, backbone_dim))
            elif fusion_method == "attentive":
                # Attentive pooler for slice fusion
                self.slice_fusion = AttentivePooler(
                    num_queries=1,
                    embed_dim=backbone_dim,
                    num_heads=16 if backbone_dim % 16 == 0 else 12,
                    mlp_ratio=4.0,
                    depth=fusion_depth,
                    qkv_bias=True,
                    complete_block=True,
                    use_activation_checkpointing=False,  # Will be controlled by set_grad_checkpointing
                )
            else:
                raise ValueError(f"Unknown fusion method: {fusion_method}")
            
            self.projection = nn.Linear(backbone_dim, output_dim)
            
        def forward(self, x):  # x: (B, 3, D, H, W) - CT volumes
            B, _, D, H, W = x.shape
            
            # Reshape for slice processing: (B, 3, D, H, W) -> (B*D, 3, H, W)
            x = rearrange(x, 'b c d h w -> (b d) c h w')
            
            # Vectorized slice-by-slice normalization to [0, 1] - all 3 channels are identical
            x_flat = x.reshape(x.shape[0], -1)  # (B*D, 3*H*W)
            slice_min = x_flat.min(dim=1, keepdim=True)[0]  # (B*D, 1)
            slice_max = x_flat.max(dim=1, keepdim=True)[0]  # (B*D, 1)
            # Use torch.where for cleaner vectorization:
            range_vals = slice_max - slice_min  # (B*D, 1)
            safe_range = torch.where(range_vals < 1e-5, torch.ones_like(range_vals), range_vals)
            x_norm = (x_flat - slice_min) / safe_range
            x = x_norm.reshape(B*D, 3, H, W)  # (B*D, 3, H, W) - use calculated dimensions
            
            # Apply ImageNet normalization (DINOv2 expectation)
            imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
            x = (x - imagenet_mean) / imagenet_std
            
            # Process each slice through DINOv2
            if self.use_gradient_checkpointing and self.training:
                # Process slices in chunks with gradient checkpointing
                def process_slice_batch(x_batch):
                    return self.backbone(x_batch)
                
                # Use gradient checkpointing for memory efficiency
                slice_features = torch.utils.checkpoint.checkpoint(
                    process_slice_batch, x, use_reentrant=False
                )
            else:
                slice_features = self.backbone(x)  # (B*D, backbone_dim)
            
            slice_features = rearrange(slice_features, '(b d) e -> b d e', b=B)  # (B, D, backbone_dim)
            
            if self.fusion_method == "transformer":
                # Add CLS token and apply transformer fusion
                cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, backbone_dim)
                tokens = torch.cat([slice_features, cls_tokens], dim=1)  # (B, D+1, backbone_dim)
                
                # Transformer fusion with x_transformers
                if self.use_gradient_checkpointing and self.training:
                    # Use gradient checkpointing for transformer layers
                    fused = torch.utils.checkpoint.checkpoint(
                        self.slice_fusion, tokens, use_reentrant=False
                    )
                else:
                    fused = self.slice_fusion(tokens)  # (B, D+1, backbone_dim)
                volume_features = fused[:, -1]  # Use CLS token: (B, backbone_dim)
            elif self.fusion_method == "attentive":
                # Attentive pooling over slice features
                pooled = self.slice_fusion(slice_features)  # (B, 1, backbone_dim)
                volume_features = pooled.squeeze(1)  # (B, backbone_dim)
            
            return self.projection(volume_features)
        
        def set_grad_checkpointing(self, enable=True):
            """Enable or disable gradient checkpointing for memory efficiency."""
            self.use_gradient_checkpointing = enable
            
            # Enable checkpointing for DinoV2/V3 backbone if it supports it
            if hasattr(self.backbone, 'set_grad_checkpointing'):
                self.backbone.set_grad_checkpointing(enable)
            elif hasattr(self.backbone, 'gradient_checkpointing_enable'):
                if enable:
                    self.backbone.gradient_checkpointing_enable()
                else:
                    # Some models don't have disable method, set the flag directly
                    if hasattr(self.backbone, 'gradient_checkpointing'):
                        self.backbone.gradient_checkpointing = False
            
            # Enable checkpointing for fusion module
            if self.fusion_method == "transformer":
                # x_transformers Encoder doesn't have built-in checkpointing,
                # we handle it manually in forward pass
                pass
            elif self.fusion_method == "attentive":
                # Update AttentivePooler's checkpointing flag
                if hasattr(self.slice_fusion, 'use_activation_checkpointing'):
                    self.slice_fusion.use_activation_checkpointing = enable
            
            print(f"Gradient checkpointing {'enabled' if enable else 'disabled'} for DinoV2Visual3D")
        
        @property
        def conv1(self):
            # Dummy property for dtype compatibility
            return self.projection

    # Replace visual encoder with 3D version
    model.visual = DinoV2Visual3D(dinov2_backbone, backbone_dim, params['embed_dim'], 
                                   fusion_method=fusion_method, fusion_depth=fusion_depth)
    if dino_version == "v3":
        model_variant = "vitl16" if "vitl" in dinov2_model_name.lower() else "vitb16"
        dino_name = f"DinoV3 ({model_variant})"
    else:
        dino_name = f"DinoV2 ({dinov2_model_name})"
    print(f"Loaded CLIP model with 3D {dino_name} vision encoder")
    if fusion_method == "transformer":
        print(f"Using x_transformers with flash attention and rotary embeddings (depth={fusion_depth})")
    elif fusion_method == "attentive":
        print(f"Using attentive pooler for slice fusion (depth={fusion_depth})")
    
    # Freeze backbone if requested
    if freeze_dinov2:
        for param in model.visual.backbone.parameters():
            param.requires_grad = False
        print("DinoV2 backbone frozen")
    
    # if a model_path is provided, load in weights to backbone
    if model_path != None: 
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Move entire model to device
    model = model.to(device)
    return model
    
    
def preprocess_text(texts, model):        
    _tokenizer = SimpleTokenizer()
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), model.context_length, dtype=torch.long)
    
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > model.context_length:
            tokens = tokens[:model.context_length]
            tokens[model.context_length - 1] = eot_token
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result


def setup_validation(config, num_workers=2):
    """
    FUNCTION: setup_validation
    ---------------------------------
    This function sets up 3D CT validation data for training monitoring.
    
    args:
        * config - configuration object with validation parameters
    
    Returns validation loader, ground truth labels, label names, templates, and input resolution
    """
    # Default validation file paths
    val_ct_filepath = getattr(config, 'val_ct_filepath', 'data/ct_volumes.h5')
    val_label_path = getattr(config, 'val_label_path', 'data/ct_rate/valid_predicted_labels.csv')
    val_batch_size = getattr(config, 'val_batch_size', 4)
    
    # Check if validation files exist
    if not os.path.exists(val_ct_filepath) or not os.path.exists(val_label_path):
        print("Warning: Validation files not found. Skipping validation setup.")
        return None, None, None, None, None
    
    # CT-specific labels from the validation CSV
    val_labels = ['Medical material', 'Arterial wall calcification', 'Cardiomegaly', 
                  'Pericardial effusion', 'Coronary artery wall calcification', 'Hiatal hernia',
                  'Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule',
                  'Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion',
                  'Mosaic attenuation pattern', 'Peribronchial thickening', 'Consolidation',
                  'Bronchiectasis', 'Interlobular septal thickening']

    # Define standard +/- templates for softmax evaluation
    val_templates = [("{}", "no {}")]  # Using a tuple pair for softmax eval

    try:
        # Load ground truth validation labels
        print(f"Loading validation labels from: {val_label_path}")
        val_df = pd.read_csv(val_label_path)
        
        # Extract ground truth labels (excluding VolumeName column)
        y_true_val = val_df[val_labels].values  # Shape: (N_samples, N_labels)
        volume_names = val_df['VolumeName'].values
        
        print(f"Loaded {len(y_true_val)} validation samples with {len(val_labels)} labels")

        # For 3D CT validation, create a simple CT-only dataset
        print("Setting up 3D CT validation...")
        
        class CTValidationDataset(data.Dataset):
            """Validation dataset for CT volumes only."""
            def __init__(self, img_path, volume_names):
                self.img_dset = h5py.File(img_path, 'r')['ct_volumes']
                self.num_volumes = len(volume_names)
                
                # Simple positional alignment - assume HDF5 and CSV are in same order
                print(f"Validation dataset: {self.img_dset.shape[0]} volumes in HDF5, {self.num_volumes} in labels CSV")
                assert self.img_dset.shape[0] >= self.num_volumes, f"HDF5 has fewer volumes ({self.img_dset.shape[0]}) than labels ({self.num_volumes})"
                
            def __len__(self):
                return self.num_volumes
            
            def __getitem__(self, idx):
                img = self.img_dset[idx]  # (D, H, W)
                img = np.expand_dims(img, axis=0)  # Add channel: (1, D, H, W)
                img = np.repeat(img, 3, axis=0)  # Repeat for RGB: (3, D, H, W)
                img = torch.from_numpy(img).float()
                return {'img': img, 'idx': idx}
        
        val_dataset = CTValidationDataset(val_ct_filepath, volume_names)
        input_resolution = 224  # Fixed for CT processing

        # Create validation dataloader
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,  # No need to shuffle for validation
            num_workers=num_workers,
            pin_memory=True
        )

        print("Validation setup complete.")
        return val_loader, y_true_val, val_labels, val_templates, input_resolution
    
    except Exception as e:
        print(f"Warning: Failed to setup validation: {e}")
        return None, None, None, None, None


def make(config, ct_filepath, txt_filepath, model_path=None, num_workers=2, local_rank=0, rank=0, use_ddp=False, world_size=1): 
    '''
    FUNCTION: make
    ---------------------------------
    This function makes the model, the data loader, loss and optimizer for 3D CT training.
    
    args: 
        * config - dict, configuration of experiment
        * ct_filepath - string, filepath to CT volumes
        * txt_filepath - string, filepath to corresponding text reports
        * model_path - string, filepath to previously trained model
        * local_rank - int, local GPU rank for device assignment
        * rank - int, global rank for distributed training
        * use_ddp - bool, whether to use DistributedDataParallel
        * world_size - int, total number of processes for distributed training
    '''
    data_loader, device, sampler = load_data(ct_filepath, txt_filepath, batch_size=config.batch_size, column=config.column, num_workers=num_workers, local_rank=local_rank, rank=rank, use_ddp=use_ddp, world_size=world_size)
    
    model = load_clip(model_path=model_path, context_length=config.context_length, 
                      dinov2_model_name=getattr(config, 'dinov2_model_name', 'dinov2_vitb14'),
                      dino_version=getattr(config, 'dino_version', 'v2'),
                      freeze_dinov2=getattr(config, 'freeze_dinov2', False), 
                      local_rank=local_rank,
                      fusion_method=getattr(config, 'fusion_method', 'transformer'),
                      fusion_depth=getattr(config, 'fusion_depth', 4))
    model.to(device)
    print('Model on Device.')

    # Select loss function based on config
    loss_type = getattr(config, 'loss_type', 'clip')
    
    if loss_type == 'siglip':
        # SigLIP loss from OpenCLIP - better for small batches, works even with batch_size=1
        from loss import SigLIPLoss
        criterion = SigLIPLoss(
            cache_labels=getattr(config, 'cache_labels', False),
            rank=rank,
            world_size=world_size,
        ).to(device)
        print(f"Using SigLIP loss (better for small batches)")
        
    elif loss_type == 'clip_optimized':
        # Optimized CLIP loss with memory-efficient techniques from OpenCLIP
        from loss import CLIPLossOptimized
        criterion = CLIPLossOptimized(
            local_loss=getattr(config, 'local_loss', False),
            gather_with_grad=getattr(config, 'gather_with_grad', False),
            cache_labels=getattr(config, 'cache_labels', False),
            rank=rank,
            world_size=world_size,
        ).to(device)
        print(f"Using optimized CLIP loss (local_loss={criterion.local_loss}, gather_with_grad={criterion.gather_with_grad})")
    else:
        # Standard CLIP loss - requires larger batches for good performance
        criterion = nn.CrossEntropyLoss().to(device)
        print(f"Using standard CLIP loss (CrossEntropy)")
    
    # Check if we should use parameter groups
    use_param_groups = getattr(config, 'use_param_groups', False)
    
    if use_param_groups:
        # Create parameter groups with different learning rates
        backbone_lr = config.lr * getattr(config, 'backbone_lr_factor', 0.1)
        backbone_wd = getattr(config, 'backbone_wd', 0.05)
        
        # Group 1: DinoV2 backbone (pre-trained, lower LR and weight decay)
        backbone_params = []
        # Group 2: All other parameters (new/scratch, standard LR and weight decay)
        other_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'visual.backbone' in name:
                    backbone_params.append(param)
                else:
                    other_params.append(param)
        
        # Create optimizer with parameter groups
        param_groups = [
            {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': backbone_wd, 'name': 'dinov2_backbone'},
            {'params': other_params, 'lr': config.lr, 'weight_decay': config.weight_decay, 'name': 'other'}
        ]
        
        optimizer = optim.AdamW(param_groups)
        
        print(f'Created optimizer with parameter groups:')
        print(f'  - DinoV2 backbone: {len(backbone_params)} params, lr={backbone_lr}, wd={backbone_wd}')
        print(f'  - Other components: {len(other_params)} params, lr={config.lr}, wd={config.weight_decay}')
    else:
        # Original optimizer setup
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        print(f'Created standard AdamW optimizer with lr={config.lr}')
    
    return model, data_loader, device, criterion, optimizer, sampler

