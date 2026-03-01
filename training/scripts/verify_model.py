"""Verify Ministral 3 8B model architecture for VLA conversion.

This script loads the model, prints its structure, and confirms
dimensions needed for the spatial decoder and forward mods.

Usage::

    cd training
    python scripts/verify_model.py

    # With 4-bit quantization (saves VRAM)
    python scripts/verify_model.py --quantize

    # Config-only mode (no model download — just verify config)
    python scripts/verify_model.py --config-only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Make training/src importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _register_ministral3_if_needed():
    """Register the ministral3 text model type if not already known.

    The Ministral-3-8B multimodal config references ``ministral3`` as the
    text model type, but some transformers versions don't have it in
    CONFIG_MAPPING.  We map it to ``mistral`` which is functionally
    equivalent for loading purposes.
    """
    try:
        from transformers import AutoConfig
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        if "ministral3" not in CONFIG_MAPPING:
            # Map ministral3 → mistral (same architecture, different name)
            from transformers import MistralConfig

            CONFIG_MAPPING.register("ministral3", MistralConfig, exist_ok=True)
            logger.info("Registered 'ministral3' → MistralConfig in CONFIG_MAPPING")
        else:
            logger.info("'ministral3' already in CONFIG_MAPPING")
    except Exception as e:
        logger.warning("Could not register ministral3: %s", e)
        # Try alternate approach
        try:
            from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

            if "ministral3" not in CONFIG_MAPPING_NAMES:
                CONFIG_MAPPING_NAMES["ministral3"] = "MistralConfig"
                logger.info("Registered 'ministral3' via CONFIG_MAPPING_NAMES")
        except Exception as e2:
            logger.warning("Fallback registration also failed: %s", e2)


def _test_spatial_decoder():
    """Test SpatialDecoder and ActionHead forward pass + gradient flow."""
    from src.forward_mods.spatial_decoder import (
        ActionHead,
        SpatialDecoder,
        SpatialDecoderLayer,
    )

    print("\n  --- Spatial Decoder Test ---")

    # Test SpatialDecoderLayer
    layer = SpatialDecoderLayer(hidden_dim=4096, num_heads=32, intermediate_dim=4096)
    dummy_input = torch.randn(1, 128, 4096)
    output = layer(dummy_input)
    assert isinstance(output, tuple)
    assert output[0].shape == (1, 128, 4096)
    print(f"    SpatialDecoderLayer: input {dummy_input.shape} → output {output[0].shape} ✓")

    # Test ActionHead
    head = ActionHead(hidden_dim=4096, action_dim=3, action_hidden_dim=256)
    dummy_hidden = torch.randn(2, 128, 4096, requires_grad=True)
    mean, log_std = head(dummy_hidden)
    assert mean.shape == (2, 3), f"Expected (2, 3), got {mean.shape}"
    assert log_std.shape == (2, 3), f"Expected (2, 3), got {log_std.shape}"
    print(f"    ActionHead: hidden {dummy_hidden.shape} → mean {mean.shape}, log_std {log_std.shape} ✓")

    # Verify mean is in [0, 1] (Sigmoid)
    assert mean.min() >= 0.0 and mean.max() <= 1.0, f"Mean should be in [0,1], got [{mean.min()}, {mean.max()}]"
    print(f"    ActionHead mean range: [{mean.min():.4f}, {mean.max():.4f}] (should be [0, 1]) ✓")

    # Test gradient flow
    loss = mean.sum() + log_std.sum()
    loss.backward()
    assert dummy_hidden.grad is not None, "Gradient did not flow back to hidden states"
    assert dummy_hidden.grad.abs().sum() > 0, "Gradient is all zeros"
    print(f"    Gradient flow through ActionHead: ✓ (grad norm = {dummy_hidden.grad.norm():.6f})")

    # Test with pool_mask
    pool_mask = torch.zeros(2, 128, dtype=torch.bool)
    pool_mask[:, :64] = True
    dummy_hidden2 = torch.randn(2, 128, 4096)
    mean2, log_std2 = head(dummy_hidden2, pool_mask=pool_mask)
    assert mean2.shape == (2, 3)
    print(f"    ActionHead with pool_mask: ✓")

    # Count parameters
    head_params = sum(p.numel() for p in head.parameters())
    layer_params = sum(p.numel() for p in layer.parameters())
    print(f"    ActionHead parameters: {head_params:,}")
    print(f"    SpatialDecoderLayer parameters: {layer_params:,}")
    print(f"    Spatial decoder test PASSED ✓")


def _test_forward_mods_config():
    """Verify forward_mod configs load and apply correctly."""
    from src.config import ForwardModConfig

    print("\n  --- Forward Mod Config Test ---")

    # Text baseline: no mods
    cfg_text = ForwardModConfig(enabled=False, mode="text")
    assert cfg_text.names == [], f"Expected empty names, got {cfg_text.names}"
    print(f"    Text baseline: names={cfg_text.names} ✓")

    # Action full: all mods
    cfg_full = ForwardModConfig(
        enabled=True, mode="action",
        temporal_position=True, cross_frame_attention=True, spatial_decoder=True,
    )
    expected = ["temporal_position", "cross_frame_attention", "spatial_decoder"]
    assert cfg_full.names == expected, f"Expected {expected}, got {cfg_full.names}"
    print(f"    Action full: names={cfg_full.names} ✓")

    print(f"    Forward mod config test PASSED ✓")


def main():
    parser = argparse.ArgumentParser(description="Verify Ministral 3 8B architecture")
    parser.add_argument(
        "--model-name",
        type=str,
        default="mistralai/Ministral-3-8B-Instruct-2512-BF16",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Load in 4-bit (QLoRA-ready)",
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Only verify config and spatial decoder (no model download)",
    )
    args = parser.parse_args()

    model_name = args.model_name

    # Always test spatial decoder (no model needed)
    _test_spatial_decoder()
    _test_forward_mods_config()

    if args.config_only:
        print("\n  --- Config-only mode: skipping model download ---")
        print("\n" + "=" * 80)
        print("  CONFIG VERIFICATION COMPLETE")
        print("=" * 80 + "\n")
        return

    # Register ministral3 before any model loading
    _register_ministral3_if_needed()

    # ---- 1. Load processor ----
    logger.info("Loading processor from %s ...", model_name)
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # ---- 2. Load model ----
    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
        "attn_implementation": "sdpa",  # safe on all platforms
    }

    if args.quantize:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["quantization_config"] = bnb_config

    logger.info("Loading model from %s ...", model_name)
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(model_name, **load_kwargs)

    # ---- 3. Print top-level architecture ----
    print("\n" + "=" * 80)
    print("  MODEL ARCHITECTURE REPORT")
    print("=" * 80)

    print(f"\n  Model class: {model.__class__.__name__}")
    print(f"  Model config class: {model.config.__class__.__name__}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,} ({total_params / 1e9:.2f}B)")

    # ---- 4. Top-level modules ----
    print("\n  --- Top-level modules ---")
    for name, child in model.named_children():
        param_count = sum(p.numel() for p in child.parameters())
        print(f"    {name}: {child.__class__.__name__} ({param_count:,} params)")

    # ---- 5. Vision tower details ----
    print("\n  --- Vision Tower ---")
    vision_tower = None
    for attr in ["vision_tower", "vision_model", "visual"]:
        if hasattr(model, attr):
            vision_tower = getattr(model, attr)
            print(f"    Found as model.{attr}: {vision_tower.__class__.__name__}")
            break

    if vision_tower is None:
        for name, mod in model.named_modules():
            if "vision" in name.lower() and hasattr(mod, "config"):
                vision_tower = mod
                print(f"    Found at {name}: {mod.__class__.__name__}")
                break

    if vision_tower is not None:
        vt_params = sum(p.numel() for p in vision_tower.parameters())
        print(f"    Vision tower params: {vt_params:,} ({vt_params / 1e6:.1f}M)")
        if hasattr(vision_tower, "config"):
            vc = vision_tower.config
            for attr_name in [
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "image_size",
                "patch_size",
                "num_channels",
            ]:
                if hasattr(vc, attr_name):
                    print(f"    config.{attr_name} = {getattr(vc, attr_name)}")

    # ---- 6. Multimodal projector ----
    print("\n  --- Multimodal Projector ---")
    projector = None
    for attr in [
        "multi_modal_projector",
        "mm_projector",
        "projector",
        "connector",
    ]:
        if hasattr(model, attr):
            projector = getattr(model, attr)
            print(f"    Found as model.{attr}: {projector.__class__.__name__}")
            break

    if projector is not None:
        proj_params = sum(p.numel() for p in projector.parameters())
        print(f"    Projector params: {proj_params:,}")
        for name, mod in projector.named_modules():
            if isinstance(mod, torch.nn.Linear):
                print(
                    f"      {name}: Linear({mod.in_features}, {mod.out_features})"
                )

    # ---- 7. Language model details ----
    print("\n  --- Language Model ---")
    lm = None
    for attr in ["language_model", "model", "lm"]:
        candidate = getattr(model, attr, None)
        if candidate is not None and hasattr(candidate, "layers") or (
            hasattr(candidate, "model") and hasattr(candidate.model, "layers")
        ):
            lm = candidate
            print(f"    Found as model.{attr}: {lm.__class__.__name__}")
            break

    if lm is None:
        for name, mod in model.named_children():
            if hasattr(mod, "model") and hasattr(mod.model, "layers"):
                lm = mod
                print(f"    Found at model.{name}: {mod.__class__.__name__}")
                break
            elif hasattr(mod, "layers"):
                lm = mod
                print(f"    Found at model.{name}: {mod.__class__.__name__}")
                break

    if lm is not None:
        lm_params = sum(p.numel() for p in lm.parameters())
        print(f"    LM params: {lm_params:,} ({lm_params / 1e9:.2f}B)")

        layers = None
        if hasattr(lm, "layers"):
            layers = lm.layers
        elif hasattr(lm, "model") and hasattr(lm.model, "layers"):
            layers = lm.model.layers

        if layers is not None:
            num_layers = len(layers)
            print(f"    Number of transformer layers: {num_layers}")

            first_layer = layers[0]
            print(f"    Layer 0 class: {first_layer.__class__.__name__}")
            for name, child in first_layer.named_children():
                print(f"      {name}: {child.__class__.__name__}")
                if isinstance(child, torch.nn.Linear):
                    print(
                        f"        -> Linear({child.in_features}, {child.out_features})"
                    )
                for sub_name, sub_child in child.named_children():
                    if isinstance(sub_child, torch.nn.Linear):
                        print(
                            f"        {sub_name}: Linear({sub_child.in_features}, {sub_child.out_features})"
                        )

            last_layer = layers[-1]
            print(f"\n    Layer {num_layers - 1} class: {last_layer.__class__.__name__}")

        lm_head = None
        if hasattr(lm, "lm_head"):
            lm_head = lm.lm_head
        elif hasattr(model, "lm_head"):
            lm_head = model.lm_head

        if lm_head is not None:
            print(f"\n    LM head: {lm_head.__class__.__name__}")
            if isinstance(lm_head, torch.nn.Linear):
                print(
                    f"      Linear({lm_head.in_features}, {lm_head.out_features})"
                )
                print(f"      -> hidden_dim = {lm_head.in_features}")
                print(f"      -> vocab_size = {lm_head.out_features}")

        if hasattr(lm, "config"):
            lm_config = lm.config
        elif hasattr(model, "config") and hasattr(model.config, "text_config"):
            lm_config = model.config.text_config
        else:
            lm_config = model.config

        print("\n    LM config attributes:")
        for attr_name in [
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
            "vocab_size",
            "rms_norm_eps",
            "rope_theta",
            "sliding_window",
        ]:
            if hasattr(lm_config, attr_name):
                print(f"      {attr_name} = {getattr(lm_config, attr_name)}")

    # ---- 8. All linear modules summary ----
    print("\n  --- All Linear layer shapes (unique) ---")
    linear_shapes: dict[str, set] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            shape = f"({mod.in_features}, {mod.out_features})"
            short = name.rsplit(".", 1)[-1]
            if short not in linear_shapes:
                linear_shapes[short] = set()
            linear_shapes[short].add(shape)

    for short, shapes in sorted(linear_shapes.items()):
        for s in sorted(shapes):
            print(f"    {short}: {s}")

    # ---- 9. Full named modules path dump (first 100) ----
    print("\n  --- Full module path dump (top 100) ---")
    all_modules = list(model.named_modules())
    for i, (name, mod) in enumerate(all_modules[:100]):
        if name:
            print(f"    {name}: {mod.__class__.__name__}")

    if len(all_modules) > 100:
        print(f"    ... ({len(all_modules) - 100} more modules)")

    # ---- 10. LoRA target check ----
    print("\n  --- LoRA target modules ---")
    proj_modules = set()
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            short = name.rsplit(".", 1)[-1]
            if short.endswith("_proj"):
                proj_modules.add(short)
    print(f"    Available *_proj modules: {sorted(proj_modules)}")

    # ---- 11. Apply LoRA and count trainable params ----
    print("\n  --- LoRA Application ---")
    try:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    except Exception as e:
        logger.warning("LoRA application failed: %s", e)

    # ---- 12. Memory usage ----
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"\n  --- VRAM Usage ---")
        print(f"    Allocated: {allocated:.2f} GB")
        print(f"    Reserved:  {reserved:.2f} GB")
    elif torch.backends.mps.is_available():
        allocated = torch.mps.current_allocated_memory() / 1e9
        print(f"\n  --- MPS Memory Usage ---")
        print(f"    Allocated: {allocated:.2f} GB")

    # ---- 13. Dummy forward pass ----
    print("\n  --- Dummy Forward Pass ---")
    try:
        from PIL import Image
        from src.utils import TOOLS

        dummy_images = [
            Image.new("RGB", (512, 512), color=(r * 60, 120, 80))
            for r in range(4)
        ]

        user_content = [{"type": "image", "image": img} for img in dummy_images]
        user_content.append({"type": "text", "text": "Shoot the duck."})
        messages = [{"role": "user", "content": user_content}]

        inputs = processor.apply_chat_template(
            messages,
            tools=TOOLS,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        print(f"    Input IDs shape: {inputs['input_ids'].shape}")
        if "pixel_values" in inputs:
            pv = inputs["pixel_values"]
            if isinstance(pv, list):
                print(f"    Pixel values: list of {len(pv)} tensors")
                for i, t in enumerate(pv):
                    print(f"      [{i}] shape={t.shape}, dtype={t.dtype}")
            else:
                print(f"    Pixel values shape: {pv.shape}")

        if "image_sizes" in inputs:
            print(f"    Image sizes: {inputs['image_sizes']}")

        with torch.no_grad():
            outputs = model(**inputs)

        print(f"    Output logits shape: {outputs.logits.shape}")
        print(f"    Forward pass SUCCEEDED")

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            print(f"    VRAM after forward: {allocated:.2f} GB")

    except Exception as e:
        logger.error("Dummy forward pass failed: %s", e, exc_info=True)

    print("\n" + "=" * 80)
    print("  VERIFICATION COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
