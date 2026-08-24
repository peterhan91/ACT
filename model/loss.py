"""
Optimized CLIP Loss with memory-efficient techniques from OpenCLIP.
CLIPLossOptimized and SigLIPLoss are adapted from mlfoundations/open_clip (MIT License).
Includes Local Loss, Gather with Gradient, and support for Gradient Accumulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def gather_features(
    image_features,
    text_features,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
):
    """
    Gather features from all GPUs with optional gradient flow.
    
    Args:
        image_features: Local image features
        text_features: Local text features
        local_loss: If True, compute loss locally to reduce memory
        gather_with_grad: If True, allow gradients through gather operation
        rank: Current process rank
        world_size: Total number of processes
    """
    if world_size == 1:
        return image_features, text_features
    
    # Gather tensors from all GPUs
    if gather_with_grad:
        # Use differentiable all_gather from torch.distributed.nn
        all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
        all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
    else:
        # Standard all_gather without gradients
        gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
        gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        
        if not local_loss:
            # Ensure gradients for local rank when features don't have gradients
            gathered_image_features[rank] = image_features
            gathered_text_features[rank] = text_features
        
        all_image_features = torch.cat(gathered_image_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)
    
    return all_image_features, all_text_features


class CLIPLossOptimized(nn.Module):
    """
    Optimized CLIP loss with memory-efficient techniques.
    
    Args:
        local_loss: Compute loss locally to reduce memory complexity from O(n^2) to O(n)
        gather_with_grad: Enable gradient flow through distributed gathering
        cache_labels: Cache ground truth labels to avoid recomputation
        rank: Current process rank
        world_size: Total number of processes
    """
    
    def __init__(
        self,
        local_loss: bool = False,
        gather_with_grad: bool = False,
        cache_labels: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        
        # Cache state
        self.prev_num_logits = 0
        self.labels = {}
    
    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        """Calculate and cache ground truth labels."""
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels
    
    def get_logits(self, image_features, text_features, logit_scale):
        """Compute logits with optional distributed gathering."""
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
            )
            
            if self.local_loss:
                # Local loss: compute logits between local features and all gathered features
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                # Global loss: compute full similarity matrix
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text
    
    def forward(
        self,
        image_features,
        text_features,
        logit_scale,
        output_dict=False,
    ):
        """
        Compute CLIP loss.
        
        Args:
            image_features: Normalized image features
            text_features: Normalized text features
            logit_scale: Temperature parameter (usually model.logit_scale.exp())
            output_dict: If True, return dict with individual losses
        """
        device = image_features.device
        
        # Get logits
        logits_per_image, logits_per_text = self.get_logits(
            image_features, text_features, logit_scale
        )
        
        # Get ground truth
        labels = self.get_ground_truth(device, logits_per_image.shape[0])
        
        # Compute losses
        loss_img = F.cross_entropy(logits_per_image, labels)
        loss_txt = F.cross_entropy(logits_per_text, labels)
        total_loss = (loss_img + loss_txt) / 2
        
        if output_dict:
            return {
                "contrastive_loss": total_loss,
                "image_loss": loss_img,
                "text_loss": loss_txt,
            }
        return total_loss


class GradientAccumulator:
    """
    Helper class for gradient accumulation with cached features.
    Used for the two-pass approach in OpenCLIP.
    """
    
    def __init__(self, accum_freq: int = 1):
        self.accum_freq = accum_freq
        self.reset()
    
    def reset(self):
        """Reset accumulated features."""
        self.accum_images = []
        self.accum_texts = []
        self.accum_features = {}
    
    def cache_features(self, images, texts, model_output):
        """
        Cache features without gradients for accumulation.
        
        Args:
            images: Input images
            texts: Input texts
            model_output: Dict with model outputs (features, logit_scale, etc.)
        """
        # Store inputs
        self.accum_images.append(images)
        self.accum_texts.append(texts)
        
        # Cache features (excluding logit_scale and other scalars)
        for key, val in model_output.items():
            if key not in ["logit_scale", "logit_bias"]:
                if key in self.accum_features:
                    self.accum_features[key].append(val)
                else:
                    self.accum_features[key] = [val]
    
    def compute_accumulated_loss(
        self,
        idx: int,
        model,
        loss_fn,
        device,
        autocast_dtype=torch.float16,
    ):
        """
        Compute loss for accumulated batch using cached features.
        
        Args:
            idx: Index of current sub-batch
            model: CLIP model
            loss_fn: Loss function
            device: Device to run on
            autocast_dtype: Dtype for autocast
        """
        images = self.accum_images[idx]
        texts = self.accum_texts[idx]
        
        with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype):
            # Handle DDP module wrapper
            model_forward = model.module if hasattr(model, 'module') else model
            
            # Forward pass with gradients
            image_features = model_forward.encode_image(images)
            text_features = model_forward.encode_text(texts)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Build inputs for loss with cached features from other batches
            all_image_features = []
            all_text_features = []
            
            for j in range(len(self.accum_images)):
                if j == idx:
                    # Use current batch with gradients
                    all_image_features.append(image_features)
                    all_text_features.append(text_features)
                else:
                    # Use cached features without gradients
                    all_image_features.append(self.accum_features["image_features"][j])
                    all_text_features.append(self.accum_features["text_features"][j])
            
            # Concatenate all features
            all_image_features = torch.cat(all_image_features, dim=0)
            all_text_features = torch.cat(all_text_features, dim=0)
            
            # Compute loss
            logit_scale = model_forward.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ all_text_features.T
            logits_per_text = logit_scale * text_features @ all_image_features.T
            
            # Labels for the current batch position
            batch_size = images.shape[0]
            labels = torch.arange(batch_size, device=device) + idx * batch_size
            
            loss_img = F.cross_entropy(logits_per_image, labels)
            loss_txt = F.cross_entropy(logits_per_text, labels)
            loss = (loss_img + loss_txt) / 2
        
        return loss
    
    def should_accumulate(self, step: int) -> bool:
        """Check if we should accumulate at this step."""
        return (step + 1) % self.accum_freq != 0


class SigLIPLoss(nn.Module):
    """
    Sigmoid Loss for Language Image Pre-Training (SigLIP) from OpenCLIP
    https://arxiv.org/abs/2303.15343
    
    This implementation is based on OpenCLIP's SigLipLoss class.
    """
    
    def __init__(
        self,
        cache_labels: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        
        # cache state
        self.prev_num_logits = 0
        self.labels = {}
    
    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        """
        Create ground truth labels for SigLIP loss.
        Returns a matrix of -1s with +1s on the diagonal (unless negative_only).
        """
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels
    
    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        """Compute logits between image and text features."""
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits
    
    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        """Compute SigLIP loss for a batch."""
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss
    
    def forward(
        self,
        image_features,
        text_features,
        logit_scale,
        logit_bias=None,
        output_dict=False,
    ):
        """
        Compute SigLIP loss.
        
        Args:
            image_features: Normalized image features [batch_size, dim]
            text_features: Normalized text features [batch_size, dim]
            logit_scale: Temperature parameter (usually model.logit_scale.exp())
            logit_bias: Optional bias parameter
            output_dict: If True, return dict with loss
        """
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)
        
        # For distributed training with multiple GPUs
        if self.world_size > 1:
            # We use the simple gather approach for distributed training
            # More complex implementations (bidir, shift, reduce) can be added if needed
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                local_loss=False,  # SigLIP doesn't use local loss
                gather_with_grad=True,  # Allow gradients through gather
                rank=self.rank,
                world_size=self.world_size,
            )
            
            # Compute loss with all gathered features
            loss = self._loss(all_image_features, all_text_features, logit_scale, logit_bias)
        
        return {"contrastive_loss": loss} if output_dict else loss