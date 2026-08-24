import os
import argparse
import math
from math import pi
from tqdm import tqdm
from datetime import timedelta

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.nn.functional import all_gather

from eval import evaluate
from train import make, preprocess_text, setup_validation

def parse_args():
    parser = argparse.ArgumentParser()
    # Data paths
    parser.add_argument('--ct_filepath', type=str, nargs='+', default=['/path/to/data/ctrate_train.h5'],
                       help='Path(s) to HDF5 file(s) containing CT volumes. Can specify multiple files.')
    parser.add_argument('--txt_filepath', type=str, nargs='+', default=[os.environ.get('ACT_REPO', '/path/to/ACT') + '/model/data/ct_rate/train_reports.csv'],
                       help='Path(s) to CSV file(s) containing text reports. Must match ct_filepath order.')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.2)
    parser.add_argument('--warmup_steps', type=int, default=250)
    
    # Parameter group optimization
    parser.add_argument('--use_param_groups', action='store_true', 
                       help='Use different learning rates for pre-trained vs new components')
    parser.add_argument('--backbone_lr_factor', type=float, default=0.1,
                       help='Learning rate factor for pre-trained DinoV2 backbone (lr * factor)')
    parser.add_argument('--backbone_wd', type=float, default=0.05,
                       help='Weight decay for pre-trained DinoV2 backbone (default: 0.05)')
    
    # Model parameters
    parser.add_argument('--context_length', type=int, default=77)
    parser.add_argument('--dinov2_model_name', type=str, default='dinov2_vitb14')
    parser.add_argument('--dino_version', type=str, default='v2', choices=['v2', 'v3'],
                        help='Choose between DinoV2 and DinoV3')
    parser.add_argument('--freeze_dinov2', action='store_true')
    parser.add_argument('--model_name', type=str, default="ct-clip-v1.0")
    parser.add_argument('--loss_type', type=str, default='clip', 
                        choices=['clip', 'siglip', 'clip_optimized'],
                        help='Loss function: clip (CrossEntropy), siglip (Sigmoid loss), or clip_optimized (with OpenCLIP optimizations)')
    
    # OpenCLIP optimization parameters (for clip_optimized loss)
    parser.add_argument('--local_loss', action='store_true',
                        help='Use local loss to reduce memory from O(n^2) to O(n) in distributed training')
    parser.add_argument('--gather_with_grad', action='store_true',
                        help='Enable gradient flow through distributed gathering')
    parser.add_argument('--cache_labels', action='store_true',
                        help='Cache ground truth labels to reduce memory allocations')
    parser.add_argument('--accum_freq', type=int, default=1,
                        help='Gradient accumulation frequency (effective batch = batch_size * accum_freq * world_size)')
    parser.add_argument('--grad_checkpointing', action='store_true',
                        help='Enable gradient checkpointing for memory efficiency')
    
    # Fusion method parameters
    parser.add_argument('--fusion_method', type=str, default='transformer', 
                       choices=['transformer', 'attentive'],
                       help='Slice fusion method: transformer (x_transformers) or attentive (VJEPA-style)')
    parser.add_argument('--fusion_depth', type=int, default=4,
                       help='Depth of the fusion module (number of layers)')
    
    # Torch compile optimization
    parser.add_argument('--compile', action='store_true',
                       help='Use torch.compile for faster training (requires PyTorch 2.0+)')
    parser.add_argument('--compile_mode', type=str, default='default',
                       choices=['default', 'reduce-overhead', 'max-autotune'],
                       help='Torch compile mode: default (balanced), reduce-overhead (lower memory), max-autotune (fastest)')
    
    # Validation
    parser.add_argument('--do_validate', action='store_true')
    parser.add_argument('--valid_interval', type=int, default=400)
    parser.add_argument('--val_ct_filepath', type=str, default='/path/to/data/ctrate_valid.h5')
    parser.add_argument('--val_label_path', type=str, default=os.environ.get('ACT_REPO', '/path/to/ACT') + '/model/data/ct_rate/valid_predicted_labels.csv')
    parser.add_argument('--val_batch_size', type=int, default=4)
    
    # Test dataset arguments - for final evaluation
    parser.add_argument('--test_after_training', action='store_true', help='Test on CT-rate test set after training')
    parser.add_argument('--test_ct_filepath', type=str, default='/path/to/data/ctrate_test.h5', help='CT-rate test images')
    parser.add_argument('--test_label_path', type=str, default=os.environ.get('ACT_REPO', '/path/to/ACT') + '/model/data/ct_rate/test_predicted_labels.csv', help='CT-rate test labels')
    parser.add_argument('--test_batch_size', type=int, default=4, help='Batch size for testing')
    
    # Early stopping arguments
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=5, help='Number of validation intervals to wait without improvement')
    parser.add_argument('--min_delta', type=float, default=0.001, help='Minimum change to qualify as an improvement')
    parser.add_argument('--early_stopping_metric', type=str, default='mean_auc', choices=['mean_auc', 'loss'], help='Metric to use for early stopping')
    
    # Logging and saving
    parser.add_argument('--save_dir', type=str, default="checkpoints/")
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--num_workers', type=int, default=2, help='Number of DataLoader workers for training')
    
    # DDP parameters
    parser.add_argument('--use_ddp', action='store_true')
    parser.add_argument('--backend', type=str, default='nccl')
    
    # Dummy parameters for compatibility
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--column', type=str, nargs='+', default=['Impressions_EN'],
                       help='Column name(s) in CSV containing text reports. Can be single name for all files or one per file.')
    
    args = parser.parse_args()
    return args

def setup_ddp(backend='nccl'):
    """Initialize DDP using torchrun."""
    dist.init_process_group(backend, timeout=timedelta(seconds=1800))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Use local_rank for GPU assignment
    torch.cuda.set_device(local_rank)
    print(f"DDP initialized: rank {rank}/{world_size}, local_rank {local_rank}")
    
    return local_rank, rank, world_size

def cleanup_ddp():
    """Clean up DDP."""
    dist.destroy_process_group()

def create_model_and_data(config, local_rank=0, rank=0, world_size=1):
    """Create model and data loader using the make function from train.py"""
    import torch
    # Override config for CT processing
    config.pretrained = False  # Always False for CT
    
    model, data_loader, device, criterion, optimizer, sampler = make(
        config, config.ct_filepath, config.txt_filepath, model_path=None, num_workers=config.num_workers, 
        local_rank=local_rank, rank=rank, use_ddp=config.use_ddp, world_size=world_size
    )
    
    # Enable gradient checkpointing if requested
    if getattr(config, 'grad_checkpointing', False):
        if hasattr(model, 'set_grad_checkpointing'):
            model.set_grad_checkpointing(True)
            print("Enabled gradient checkpointing")
        elif hasattr(model, 'visual') and hasattr(model.visual, 'slice_fusion'):
            # Try to enable for fusion module
            if hasattr(model.visual.slice_fusion, 'gradient_checkpointing_enable'):
                model.visual.slice_fusion.gradient_checkpointing_enable()
                print("Enabled gradient checkpointing for fusion module")
    
    # Apply torch.compile if requested (before DDP wrapping)
    if getattr(config, 'compile', False) and hasattr(torch, 'compile'):
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True  # Continue training even if compile fails
        torch._dynamo.config.cache_size_limit = 64  # Increase cache size limit to reduce recompilations
        
        compile_mode = getattr(config, 'compile_mode', 'default')
        
        # Compile model components separately for better compatibility
        try:
            # Compile visual encoder
            if hasattr(model, 'visual'):
                model.visual = torch.compile(model.visual, mode=compile_mode)
                print(f"Compiled visual encoder with mode: {compile_mode}")
            
            # Compile text encoder
            if hasattr(model, 'transformer'):
                model.transformer = torch.compile(model.transformer, mode=compile_mode)
                print(f"Compiled text encoder with mode: {compile_mode}")
            
            # Note: Don't compile the full model to avoid issues with logit_scale parameter
            print("Model components compiled with torch.compile")
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}")
            print("Continuing without compilation")
    elif getattr(config, 'compile', False):
        print("Warning: torch.compile requested but not available (requires PyTorch 2.0+)")
    
    # Wrap with DDP if needed
    if config.use_ddp:
        # Use find_unused_parameters=True for VIT-L to handle dimension differences
        find_unused = "vitl" in config.dinov2_model_name.lower()
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=find_unused)
        print(f'Model wrapped with DDP on local_rank {local_rank} (find_unused_parameters={find_unused})')
    
    # Create scheduler - adjust for gradient accumulation
    # Total optimization steps = total batches / accumulation frequency
    accum_freq = getattr(config, 'accum_freq', 1)
    total_opt_steps = (config.epochs * len(data_loader)) // accum_freq
    
    if rank == 0:
        print(f"Training configuration:")
        print(f"  - Batches per epoch: {len(data_loader)}")
        print(f"  - Gradient accumulation: {accum_freq}")
        print(f"  - Batch count per epoch: {len(data_loader) // accum_freq}")
        print(f"  - Optimization steps per epoch: {len(data_loader) // accum_freq}")
        print(f"  - Total optimization steps: {total_opt_steps}")
        print(f"  - Warmup steps: {config.warmup_steps}")
        if config.do_validate:
            print(f"  - Validation every {config.valid_interval} batch_count steps")
    
    def lr_lambda(current_step):
        if current_step < config.warmup_steps:
            return float(current_step) / float(max(1, config.warmup_steps))
        progress = float(current_step - config.warmup_steps) / float(max(1, total_opt_steps - config.warmup_steps))
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * pi))))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda")
    
    return model, data_loader, device, criterion, optimizer, scheduler, scaler, sampler

def train_epoch(model, loader, device, criterion, optimizer, scheduler, scaler, config, epoch=0, rank=0, 
                validation_state=None, sampler=None):
    """Train for one epoch with AMP and gradient accumulation."""
    model.train()
    example_ct = 0
    batch_ct = 0  # Total batches processed (raw count)
    running_loss = 0.0
    running_loss_count = 0  # Track number of losses accumulated
    step_ct = 0  # Track actual optimization steps (for scheduler)
    
    # Set epoch for distributed sampler
    if sampler is not None and hasattr(sampler, 'set_epoch'):
        sampler.set_epoch(epoch)
    
    steps_since_optimizer_step = 0  # Track if optimizer has stepped at least once
    
    # Setup for two-pass gradient accumulation if enabled
    accum_freq = getattr(config, 'accum_freq', 1)
    if accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}
    
    for i, data in enumerate(tqdm(loader, disable=(rank != 0))):
        i_accum = i // accum_freq  # Following OpenCLIP: integer division for step count
        images = data['img'].to(device)  # (B, 3, D, H, W)
        model_for_text = model.module if hasattr(model, 'module') else model
        texts = preprocess_text(data['txt'], model_for_text).to(device)
        
        model_forward = model.module if hasattr(model, 'module') else model
        
        # OpenCLIP calls zero_grad at start of every iteration (line 97)
        optimizer.zero_grad()
        
        # Two-pass gradient accumulation for better contrastive learning
        if accum_freq == 1:
            # Standard single-batch training
            with torch.amp.autocast('cuda'):
                # Get features from model
                image_features = model_forward.encode_image(images)
                text_features = model_forward.encode_text(texts)
                
                # Handle different loss types
                if config.loss_type == 'siglip':
                    # SigLIP loss from OpenCLIP - normalize features and pass logit_scale
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    logit_scale = model_forward.logit_scale.exp()
                    
                    # SigLIP handles distributed gathering internally if world_size > 1
                    loss = criterion(image_features, text_features, logit_scale)
                    
                elif config.loss_type == 'clip_optimized':
                    # Optimized CLIP loss handles normalization and gathering internally
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    logit_scale = model_forward.logit_scale.exp()
                    loss = criterion(image_features, text_features, logit_scale)
                    
                else:  # Standard CLIP loss
                    # Normalize features for CLIP
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    # Gather features across GPUs if using DDP (critical for CLIP)
                    if config.use_ddp:
                        if rank == 0 and batch_ct == 0:
                            print(f"Using distributed gathering for batch_size={images.size(0)} across {dist.get_world_size()} GPUs")
                        
                        all_image_features_list = all_gather(image_features)
                        all_text_features_list = all_gather(text_features)
                        all_image_features = torch.cat(all_image_features_list, dim=0)
                        all_text_features = torch.cat(all_text_features_list, dim=0)
                        
                        # Compute logits with all gathered features
                        logit_scale = model_forward.logit_scale.exp()
                        logits_per_image = logit_scale * image_features @ all_text_features.t()
                        logits_per_text = logit_scale * text_features @ all_image_features.t()
                        
                        # Labels need to account for rank offset
                        labels = torch.arange(images.size(0), device=device) + rank * images.size(0)
                    else:
                        # Normal computation without gathering
                        logit_scale = model_forward.logit_scale.exp()
                        logits_per_image = logit_scale * image_features @ text_features.t()
                        logits_per_text = logit_scale * text_features @ image_features.t()
                        labels = torch.arange(images.size(0), device=device)
                    
                    # CLIP loss computation
                    loss_img = criterion(logits_per_image, labels)
                    loss_txt = criterion(logits_per_text, labels)
                    loss = (loss_img + loss_txt) / 2
        
            # Backward pass for single batch
            scaler.scale(loss).backward()
            running_loss += loss.item()
            running_loss_count += 1
            
        else:
            # Two-pass accumulation: cache features first, then compute gradients
            # First pass: cache features without gradients
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    image_features = model_forward.encode_image(images)
                    text_features = model_forward.encode_text(texts)
                    
                    # Normalize for CLIP-based and SigLIP losses
                    if config.loss_type in ['clip', 'clip_optimized', 'siglip']:
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    # Cache features
                    if 'image_features' in accum_features:
                        accum_features['image_features'].append(image_features)
                        accum_features['text_features'].append(text_features)
                    else:
                        accum_features['image_features'] = [image_features]
                        accum_features['text_features'] = [text_features]
            
            accum_images.append(images)
            accum_texts.append(texts)
            
            # Update counters for every batch in accumulation
            batch_ct += 1
            example_ct += images.size(0)
            
            # If not at end of accumulation window, continue (following OpenCLIP)
            if ((i + 1) % accum_freq) > 0:
                continue
                
            # Second pass: compute gradients using all cached features
            optimizer.zero_grad()  # OpenCLIP calls this again before gradient computation
            for j in range(accum_freq):
                with torch.amp.autocast('cuda'):
                    # Re-encode batch j with gradients
                    image_features = model_forward.encode_image(accum_images[j])
                    text_features = model_forward.encode_text(accum_texts[j])
                    
                    # Normalize for CLIP-based and SigLIP losses
                    if config.loss_type in ['clip', 'clip_optimized', 'siglip']:
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    # Concatenate all features (with current batch having gradients)
                    all_image_features = torch.cat(
                        accum_features['image_features'][:j] + 
                        [image_features] + 
                        accum_features['image_features'][j+1:]
                    )
                    all_text_features = torch.cat(
                        accum_features['text_features'][:j] + 
                        [text_features] + 
                        accum_features['text_features'][j+1:]
                    )
                    
                    # Compute loss following OpenCLIP's approach: pass FULL batch to loss function
                    logit_scale = model_forward.logit_scale.exp()
                    
                    if config.loss_type == 'siglip':
                        # SigLIP loss - pass full batch to loss function
                        loss = criterion(all_image_features, all_text_features, logit_scale)
                        
                    elif config.loss_type == 'clip_optimized':
                        # Optimized CLIP loss - pass full batch to loss function
                        loss = criterion(all_image_features, all_text_features, logit_scale)
                        
                    else:
                        # Standard CLIP loss - compute manually since we use nn.CrossEntropyLoss
                        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
                        logits_per_text = logits_per_image.t()
                        
                        # Labels for full accumulated batch
                        labels = torch.arange(all_image_features.shape[0], device=device)
                        
                        loss_img = F.cross_entropy(logits_per_image, labels)
                        loss_txt = F.cross_entropy(logits_per_text, labels)
                        loss = (loss_img + loss_txt) / 2
                    
                    # No scaling needed - gradients naturally accumulate
                    scaler.scale(loss).backward()
            
            # Track the last loss value (following OpenCLIP convention)
            running_loss += loss.item()
            running_loss_count += 1
        
        # For non-accumulation case, update counters
        if accum_freq == 1:
            batch_ct += 1
            example_ct += images.size(0)
        
        # Optimizer step (following OpenCLIP timing)
        if accum_freq == 1 or (i + 1) % accum_freq == 0:
            scaler.step(optimizer)
            scaler.update()
            step_ct += 1  # Increment optimization step counter
            steps_since_optimizer_step += 1
            # Only step scheduler after at least one optimizer step
            if scheduler and steps_since_optimizer_step > 0: 
                scheduler.step()
            
            # Reset gradient accum, if enabled (following OpenCLIP line 185)
            if accum_freq > 1:
                accum_images, accum_texts, accum_features = [], [], {}
            
            # Clamp logit_scale to prevent training instability (as in original CLIP)
            with torch.no_grad():
                if hasattr(model_forward, 'logit_scale'):
                    model_forward.logit_scale.data.clamp_(0, math.log(100))
        
        # Logging - based on i_accum (following OpenCLIP)
        batch_count = i_accum + 1
        if rank == 0 and (i_accum % config.log_interval == 0 or batch_count == len(loader) // accum_freq):
            # Average loss over all sub-batches that contributed
            avg_loss = running_loss / max(1, running_loss_count)
            
            # Ensure we're using the correct GPU device for rank 0
            current_device = torch.cuda.current_device()
            gpu_mem = torch.cuda.memory_allocated(current_device) / 1024**3  # GB
            gpu_cache = torch.cuda.memory_reserved(current_device) / 1024**3  # GB
            
            # GPU utilization with pynvml
            try:
                import pynvml
                if not hasattr(pynvml, '_initialized'):
                    pynvml.nvmlInit()
                    pynvml._initialized = True
                gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(current_device)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
            except Exception as e:
                gpu_util = 0
            
            effective_batch = config.batch_size * accum_freq
            if config.use_ddp:
                effective_batch *= dist.get_world_size()
            print(f"Step {batch_count}, Batch {batch_ct}, OptStep {step_ct}, Loss: {avg_loss:.4f}, Examples: {example_ct}, "
                  f"EffBatch: {effective_batch}, GPU{current_device}: {gpu_mem:.1f}GB/{gpu_cache:.1f}GB, Util: {gpu_util}%")
            running_loss = 0.0
            running_loss_count = 0
        
        # Validation - based on i_accum steps
        early_stop_flag = False
        if config.do_validate and batch_count > 0 and batch_count % config.valid_interval == 0:
            # Synchronize all processes before validation
            if config.use_ddp:
                dist.barrier()
            
            if rank == 0:
                # Pass optimization step count for proper logging
                early_stop_flag = run_validation(model, device, config, step_ct, epoch, validation_state)
        
        # Broadcast early stopping decision across processes
        if config.use_ddp:
            stop_tensor = torch.tensor(early_stop_flag, dtype=torch.bool, device=device)
            dist.broadcast(stop_tensor, src=0)
            early_stop_flag = stop_tensor.item()
        
        if early_stop_flag:
            return batch_ct, example_ct, True
    
    # Note: With drop_last=True in DataLoader, we never have partial batches at epoch end
    # The DataLoader ensures the number of batches is always divisible by batch_size
    
    return batch_ct, example_ct, False

def run_validation(model, device, config, step, epoch, validation_state):
    """Run validation, log results, and check for early stopping."""
    model_for_val = model.module if hasattr(model, 'module') else model
    model_for_val.eval()
    
    # Initialize validation results
    mean_auc = 0
    auc_values = []
    val_labels = None
    
    # CT-RATE validation  
    val_loader, y_true_val, val_labels, val_templates, _ = setup_validation(config, num_workers=config.num_workers)
    
    if val_loader is not None:
        pos_template, neg_template = val_templates[0]
        
        # Encode text templates using clip.tokenize
        with torch.no_grad():
            pos_texts = [pos_template.format(c) for c in val_labels]
            neg_texts = [neg_template.format(c) for c in val_labels]
            import clip
            context_length = getattr(model_for_val, 'context_length', config.context_length)
            pos_tokens = clip.tokenize(pos_texts, context_length).to(device)
            neg_tokens = clip.tokenize(neg_texts, context_length).to(device)
            pos_features = model_for_val.encode_text(pos_tokens)
            neg_features = model_for_val.encode_text(neg_tokens)
            pos_features /= pos_features.norm(dim=-1, keepdim=True)
            neg_features /= neg_features.norm(dim=-1, keepdim=True)
        
        # Extract image features
        all_img_feats = []
        with torch.no_grad():
            for data in tqdm(val_loader, desc="Validation"):
                imgs = data['img'].to(device)
                feats = model_for_val.encode_image(imgs)
                feats /= feats.norm(dim=-1, keepdim=True)
                all_img_feats.append(feats.cpu())
        
        # Compute predictions
        img_feats_cat = torch.cat(all_img_feats).to(device)
        logits_pos = img_feats_cat @ pos_features.T
        logits_neg = img_feats_cat @ neg_features.T
        probs = torch.exp(logits_pos) / (torch.exp(logits_pos) + torch.exp(logits_neg))
        y_pred_val = probs.cpu().numpy()
        
        # Use ground truth labels with same positional alignment
        y_true_val_aligned = y_true_val[:len(y_pred_val)]
        
        # Evaluate
        val_results_df = evaluate(y_pred_val, y_true_val_aligned, val_labels)
        auc_cols = [col for col in val_results_df.columns if col.endswith('_auc')]
        mean_auc = val_results_df[auc_cols].mean().mean() if auc_cols else 0
        
        # Get individual AUC values for logging
        auc_values = [val_results_df[col].iloc[0] if col in val_results_df.columns else 0 for col in auc_cols]
        
        print(f"Validation at step {step}: CT-RATE Mean AUC = {mean_auc:.4f}")
    else:
        print(f"Validation at step {step}: No validation data available")
    
    # Log CT-RATE results to CSV
    with open(validation_state['val_log_path'], 'a') as f:
        f.write(f"{step},{epoch},{mean_auc:.4f},{','.join(f'{v:.4f}' for v in auc_values)}\n")
    
    # Check if this is the best model so far
    early_stop_flag = False
    if mean_auc > validation_state['best_metric'] + config.min_delta:
        validation_state['best_metric'] = mean_auc
        validation_state['best_step'] = step
        validation_state['best_epoch'] = epoch
        validation_state['intervals_without_improvement'] = 0
        
        # Save best model
        model_to_save = model.module if hasattr(model, 'module') else model
        save_model(model_to_save, validation_state['best_model_path'])
        print(f"New best model saved! CT-RATE AUC: {mean_auc:.4f} at step {step}")
    else:
        validation_state['intervals_without_improvement'] += 1
        if (config.early_stopping and 
            validation_state['intervals_without_improvement'] >= config.patience):
            print(f"Early stopping triggered! No improvement for {config.patience} validation intervals.")
            print(f"Best mean AUC: {validation_state['best_metric']:.4f} achieved at step {validation_state['best_step']} (epoch {validation_state['best_epoch']})")
            early_stop_flag = True
    
    model.train()
    return early_stop_flag

def save_model(model, path):
    """Save model state dict."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), path)

# ====================== TESTING FUNCTIONS - Added for final evaluation ======================

def find_best_model(config):
    """Find the best model saved during training."""
    model_save_dir = os.path.join(config.save_dir, config.model_name)
    
    # Check if best model exists
    best_model_path = os.path.join(model_save_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        # Read validation log to get the best AUC score
        val_log_path = os.path.join(model_save_dir, "validation_log.csv")
        if os.path.exists(val_log_path):
            try:
                import pandas as pd
                df = pd.read_csv(val_log_path)
                best_idx = df['Mean_AUC'].idxmax()
                best_auc = df.loc[best_idx, 'Mean_AUC']
                best_step = df.loc[best_idx, 'Step']
                print(f"Using best model: AUC = {best_auc:.4f} at step {best_step}")
            except:
                print("Using best model (unable to read validation log)")
        else:
            print("Using best model from training")
        return best_model_path
    
    # Fallback to final checkpoint if best model doesn't exist
    final_checkpoint = os.path.join(model_save_dir, 'final_model.pt')
    if os.path.exists(final_checkpoint):
        print("Warning: Best model not found. Using final checkpoint.")
        return final_checkpoint
    
    raise FileNotFoundError(f"No model found in {model_save_dir}")

def setup_test_dataset(test_ct_filepath, test_label_path, labels, config):
    """Setup test dataset loader and ground truth labels for CT data."""
    from train import CTDataset
    import pandas as pd
    import numpy as np
    
    print(f"Loading test labels from: {test_label_path}")
    
    # Load test labels CSV
    test_df = pd.read_csv(test_label_path)
    
    # Extract ground truth labels (exclude VolumeName column)
    label_columns = [col for col in test_df.columns if col != 'VolumeName']
    y_true_test = test_df[label_columns].values.astype(np.float32)
    
    print(f"Loading test CT data from: {test_ct_filepath}")
    print(f"Found {len(label_columns)} disease labels: {', '.join(label_columns)}")
    
    # Create test dataset (just volume names for loading)
    volume_names = test_df['VolumeName'].tolist()
    
    # Create dummy text data for test dataset (we only need volumes for testing)
    test_texts = [" "] * len(volume_names)  # Empty strings for testing
    test_dataset = CTDataset(test_ct_filepath, test_texts)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return test_loader, y_true_test, label_columns

def test_model_on_dataset(model, test_loader, y_true_test, labels, templates, device, config, dataset_name):
    """Test model on CT dataset and return results."""
    model.eval()
    context_length = getattr(model, 'context_length', config.context_length)
    pos_template, neg_template = templates[0]
    
    print(f"\\n=== Testing on {dataset_name} ===")
    
    # Encode text templates
    with torch.no_grad():
        pos_texts = [pos_template.format(c) for c in labels]
        neg_texts = [neg_template.format(c) for c in labels]
        import clip
        pos_tokens = clip.tokenize(pos_texts, context_length).to(device)
        neg_tokens = clip.tokenize(neg_texts, context_length).to(device)
        pos_features = model.encode_text(pos_tokens)
        neg_features = model.encode_text(neg_tokens)
        pos_features /= pos_features.norm(dim=-1, keepdim=True)
        neg_features /= neg_features.norm(dim=-1, keepdim=True)
    
    # Extract image features
    all_img_feats = []
    with torch.no_grad():
        for data in tqdm(test_loader, desc=f"Testing on {dataset_name}"):
            imgs = data['img'].to(device)
            feats = model.encode_image(imgs)
            feats /= feats.norm(dim=-1, keepdim=True)
            all_img_feats.append(feats.cpu())
    
    # Compute predictions and evaluate
    img_feats_cat = torch.cat(all_img_feats).to(device)
    logits_pos = img_feats_cat @ pos_features.T
    logits_neg = img_feats_cat @ neg_features.T
    probs = torch.exp(logits_pos) / (torch.exp(logits_pos) + torch.exp(logits_neg))
    y_pred_test = probs.cpu().numpy()
    
    test_results_df = evaluate(y_pred_test, y_true_test, labels)
    return test_results_df

def run_final_testing(config):
    """Run testing on CT-rate test dataset using the best model."""
    print("\\n" + "="*60)
    print("STARTING FINAL TESTING ON CT-RATE TEST DATASET")
    print("="*60)
    
    # Find best model
    best_model_path = find_best_model(config)
    
    # Load the best model
    from train import load_clip
    model = load_clip(
        model_path=best_model_path,
        context_length=config.context_length,
        dinov2_model_name=config.dinov2_model_name,
        dino_version=config.dino_version,
        fusion_method=config.fusion_method,
        fusion_depth=config.fusion_depth,
        freeze_dinov2=config.freeze_dinov2
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    results_dir = os.path.join(config.save_dir, config.model_name, "test_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Test on CT-rate test set
    if os.path.exists(config.test_ct_filepath) and os.path.exists(config.test_label_path):
        ct_rate_templates = [("{}", "no {}")]
        
        # Get labels dynamically from the test dataset
        test_loader, y_true_test, actual_labels = setup_test_dataset(
            config.test_ct_filepath, config.test_label_path, None, config)
        
        test_results = test_model_on_dataset(
            model, test_loader, y_true_test, actual_labels, 
            ct_rate_templates, device, config, "CT-rate Test")
        
        test_results.to_csv(os.path.join(results_dir, "ct_rate_test_results.csv"), index=False)
        print(f"CT-rate test results saved to: {results_dir}/ct_rate_test_results.csv")
        
        # Print overall mean AUC for all pathologies
        auc_cols = [col for col in test_results.columns if col.endswith('_auc')]
        overall_mean_auc = test_results[auc_cols].mean().mean() if auc_cols else 0
        print(f"CT-rate Overall Mean AUC ({len(auc_cols)} pathologies): {overall_mean_auc:.4f}")
        
        # Print individual AUC scores for all pathologies
        print("\\nIndividual AUC scores:")
        for col in sorted(auc_cols):
            pathology_name = col.replace('_auc', '')
            auc_score = test_results[col].iloc[0]
            print(f"  {pathology_name}: {auc_score:.4f}")
    else:
        print(f"Test files not found: {config.test_ct_filepath} or {config.test_label_path}")
    
    print("\\n" + "="*60)
    print("FINAL TESTING COMPLETED")
    print("="*60)

# ====================== END TESTING FUNCTIONS ======================

def main():
    """Main training function."""
    config = parse_args()
    
    # Validate matching lengths
    if len(config.ct_filepath) != len(config.txt_filepath):
        raise ValueError(f"Number of CT files ({len(config.ct_filepath)}) must match number of text files ({len(config.txt_filepath)})")
    
    # Setup
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    if config.use_ddp:
        local_rank, rank, world_size = setup_ddp(config.backend)
    else:
        local_rank, rank, world_size = 0, 0, 1
    
    try:
        # Create model and data
        model, data_loader, device, criterion, optimizer, scheduler, scaler, sampler = create_model_and_data(config, local_rank, rank, world_size)
        
        # Create save directory and validation state (only on rank 0)
        validation_state = None
        if rank == 0:
            save_dir = os.path.join(config.save_dir, config.model_name)
            os.makedirs(save_dir, exist_ok=True)
            
            # Initialize validation logging and state
            validation_state = {
                'val_log_path': os.path.join(save_dir, "validation_log.csv"),
                'best_model_path': os.path.join(save_dir, "best_model.pt"),
                'best_metric': float('-inf'),
                'best_step': 0,
                'best_epoch': 0,
                'intervals_without_improvement': 0
            }
            
            # Create validation log file with header (get labels from validation setup)
            if config.do_validate:
                # Try to get CT-RATE labels
                _, _, val_labels, _, _ = setup_validation(config, num_workers=config.num_workers)
                if val_labels is not None:
                    # Create header with mean AUC and individual disease labels for CT-RATE
                    disease_headers = [f"{label}_AUC" for label in val_labels]
                    header = "Step,Epoch,Mean_AUC," + ",".join(disease_headers) + "\n"
                else:
                    header = "Step,Epoch,Mean_AUC\n"
            else:
                header = "Step,Epoch,Mean_AUC\n"
            
            with open(validation_state['val_log_path'], 'w') as f:
                f.write(header)
        
        # Training loop
        early_stopped = False
        for epoch in range(config.epochs):
            print(f"\n=== Epoch {epoch+1}/{config.epochs} ===")
            batch_ct, example_ct, early_stop_flag = train_epoch(
                model, data_loader, device, criterion, optimizer, 
                scheduler, scaler, config, epoch, rank, validation_state, sampler
            )
            
            if early_stop_flag:
                early_stopped = True
                break
        
        # Save final model (only on rank 0)
        if rank == 0:
            final_path = os.path.join(save_dir, 'final_model.pt')
            model_to_save = model.module if hasattr(model, 'module') else model
            save_model(model_to_save, final_path)
            
            if early_stopped:
                print(f"Training stopped early! Final model saved to {final_path}")
            else:
                print(f"Training completed! Model saved to {final_path}")
            
            # Run final testing if requested
            if config.test_after_training:
                run_final_testing(config)
    
    finally:
        if config.use_ddp:
            cleanup_ddp()

if __name__ == "__main__":
    main()
