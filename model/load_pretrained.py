"""Load the released ACT volume-report checkpoint from Hugging Face.

Downloads clip_3d_ctrate_merlin_v1/best_model.pt from
https://huggingface.co/peterhan91/clip_3d_ct and instantiates the exact
released configuration: DINOv2 ViT-B/14 with register tokens, two-layer
Transformer slice fusion (12 heads, feed-forward multiplier 2), 768-d
projection, CLIP text tower (width 512, 12 layers, context length 77).

Usage:
    from load_pretrained import load_act
    model, preprocess_text = load_act()
"""

from huggingface_hub import hf_hub_download

from train import load_clip, preprocess_text

HF_REPO = "peterhan91/clip_3d_ct"
HF_CHECKPOINT = "clip_3d_ctrate_merlin_v1/best_model.pt"


def load_act(revision=None, device=None):
    """Return (model, preprocess_text) with the released weights loaded.

    revision: optional Hugging Face revision (commit hash or tag) to pin.
    device: optional torch device to move the model to.
    """
    ckpt = hf_hub_download(HF_REPO, HF_CHECKPOINT, revision=revision)
    model = load_clip(
        model_path=ckpt,
        context_length=77,
        dinov2_model_name="dinov2_vitb14",
        dino_version="v2",
        fusion_method="transformer",
        fusion_depth=2,
    )
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, preprocess_text


if __name__ == "__main__":
    model, _ = load_act()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded ACT clip_3d_ctrate_merlin_v1 ({n_params/1e6:.1f}M parameters)")
