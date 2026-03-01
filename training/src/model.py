"""Model loading, LoRA setup, and inference verification for Ministral-3-8B."""

from __future__ import annotations

import logging

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor

from .config import ForwardModConfig, FullConfig, LoRAConfig, ModelConfig
from .utils import TOOLS, parse_tool_call

logger = logging.getLogger(__name__)

# Dtype string → torch dtype
_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


# ---------------------------------------------------------------------------
# 2.1  Load model + processor
# ---------------------------------------------------------------------------
def load_model_and_processor(
    config: ModelConfig,
) -> tuple[AutoModelForImageTextToText, AutoProcessor]:
    """Load the Ministral-3-8B model and its processor.

    Uses ``AutoModelForImageTextToText`` which resolves to
    ``Mistral3ForConditionalGeneration`` for the mistral3 architecture
    (8.4 B LLM + 0.4 B Pixtral vision encoder).
    """
    model_name = config.model_name
    dtype = _DTYPE_MAP.get(config.torch_dtype, torch.bfloat16)

    logger.info("Loading processor from %s …", model_name)
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=config.trust_remote_code,
    )

    # GRPO / batch generation requires left-padding
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    load_kwargs: dict = {
        "torch_dtype": dtype,
        "device_map": config.device_map,
        "attn_implementation": config.attn_implementation,
        "trust_remote_code": config.trust_remote_code,
    }

    # QLoRA: 4-bit quantization
    if getattr(config, "quantize_4bit", False):
        try:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["quantization_config"] = bnb_config
            logger.info("QLoRA enabled: 4-bit NF4 quantization")
        except ImportError:
            logger.warning("bitsandbytes not available — loading without quantization")

    logger.info("Loading model from %s (dtype=%s) …", model_name, config.torch_dtype)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        **load_kwargs,
    )

    # ---- Diagnostics ----
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model architecture: %s", model.__class__.__name__)
    logger.info("Total parameters:   %.2f B", total_params / 1e9)

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        logger.info("VRAM allocated:     %.2f GB", allocated)

    return model, processor


# ---------------------------------------------------------------------------
# 2.2  Apply LoRA
# ---------------------------------------------------------------------------
def _find_target_modules(model: torch.nn.Module) -> list[str]:
    """Auto-detect linear projection layers suitable for LoRA."""
    targets: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Keep the last part of the qualified name (e.g. "q_proj")
            short = name.rsplit(".", 1)[-1]
            if short.endswith("_proj"):
                targets.add(short)
    found = sorted(targets)
    logger.info("Auto-detected LoRA target modules: %s", found)
    return found


def apply_lora(
    model: AutoModelForImageTextToText,
    lora_cfg: LoRAConfig,
) -> AutoModelForImageTextToText:
    """Wrap *model* with LoRA adapters.

    Returns the PeftModel (same reference, but with LoRA layers injected).
    """
    if not lora_cfg.enabled:
        logger.info("LoRA disabled — returning base model unchanged.")
        return model

    target_modules = lora_cfg.target_modules
    if not target_modules or target_modules == ["auto"]:
        target_modules = _find_target_modules(model)

    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=target_modules,
        bias=lora_cfg.bias,
        task_type=lora_cfg.task_type,
    )

    # Required for gradient checkpointing with LoRA
    model.enable_input_require_grads()

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model


# ---------------------------------------------------------------------------
# 2.3  Verify VLM inference  (uses native Mistral tool-calling tokens)
# ---------------------------------------------------------------------------
def test_inference(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
) -> bool:
    """Run a quick forward pass with dummy images to verify the pipeline.

    Tools are passed via ``processor.apply_chat_template(tools=…)`` so the
    tokenizer inserts ``[AVAILABLE_TOOLS]…[/AVAILABLE_TOOLS]`` and the model
    is expected to reply with ``[TOOL_CALLS] [{…}]``.
    """
    logger.info("Running inference smoke-test …")

    # 1. Dummy images (4 × 256×256 RGB)
    dummy_images = [
        Image.new("RGB", (256, 256), color=(r * 60, 120, 80))
        for r in range(4)
    ]

    # 2. Build chat messages with images + tool definitions
    user_content: list[dict] = [
        {"type": "image", "image": img} for img in dummy_images
    ]
    user_content.append(
        {
            "type": "text",
            "text": (
                "Above are 4 game frames showing duck movement. "
                "Call the shoot tool with your best prediction."
            ),
        }
    )

    messages = [{"role": "user", "content": user_content}]

    # 3. Process through the processor with native tool tokens
    inputs = processor.apply_chat_template(
        messages,
        tools=TOOLS,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 4. Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )

    # Decode new tokens (keep special tokens so we can see [TOOL_CALLS])
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    decoded_raw = processor.decode(new_tokens, skip_special_tokens=False)
    decoded_clean = processor.decode(new_tokens, skip_special_tokens=True)

    logger.info("Smoke-test raw output:\n%s", decoded_raw)

    # 5. Try parsing as a tool call
    action = parse_tool_call(decoded_raw)
    if action is not None:
        logger.info(
            "Parsed tool call: x=%.3f, y=%.3f, horizon=%d",
            action.x, action.y, action.horizon,
        )

    success = len(decoded_clean.strip()) > 0
    if success:
        logger.info("Smoke-test PASSED.")
    else:
        logger.warning("Smoke-test FAILED — empty output.")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        logger.info("VRAM after inference: %.2f GB", allocated)

    return success


# ---------------------------------------------------------------------------
# Convenience: full setup pipeline
# ---------------------------------------------------------------------------
def setup_model(config: FullConfig) -> tuple[AutoModelForImageTextToText, AutoProcessor]:
    """Load model → apply LoRA → apply forward_mods → return (model, processor).

    Forward mods are applied *after* LoRA so they can modify the
    architecture beyond LoRA's reach (e.g. replacing layers).
    """
    model, processor = load_model_and_processor(config.model)
    model = apply_lora(model, config.lora)

    # Apply forward modifications (VLA architecture changes)
    if config.forward_mod.enabled:
        from .forward_mods import apply_forward_mods

        model, applied_mods = apply_forward_mods(model, config.forward_mod)

        # Collect extra parameters from forward mods
        extra_params = []
        for mod in applied_mods:
            extra_params.extend(mod.extra_parameters())
        if extra_params:
            logger.info(
                "Forward mods added %d extra trainable parameters",
                sum(p.numel() for p in extra_params),
            )
            # Store on model for the trainer to access
            model._forward_mod_params = extra_params

    return model, processor
