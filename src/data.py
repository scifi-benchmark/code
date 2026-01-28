"""Data loading utilities for SciFi-Benchmark."""

import os
from typing import Literal

# Suppress TensorFlow/CUDA warnings before importing TF
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # 0=all, 1=no INFO, 2=no WARNING, 3=no ERROR
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # Disable oneDNN messages
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # Disable GPU (not needed for data loading)

SplitType = Literal["train", "val"]
PromptType = Literal[
    "with_constitution",
    "without_constitution",
    "with_constitution_antijailbreak",
    "with_constitution_antijailbreak_adversary",
]

# GCS bucket paths for the datasets (TFDS)
GCS_BUCKET = "gs://gresearch/robotics"
DATASET_VERSION = "0.1.0"

# HuggingFace dataset ID
HF_DATASET_ID = "sermanet/scifi-benchmark"


def load_scifi_dataset(
    split: SplitType = "val",
    prompt_type: PromptType = "with_constitution",
    source: Literal["auto", "huggingface", "tfds"] = "auto",
) -> list[dict]:
    """Load the SciFi-Benchmark dataset.

    Args:
        split: Which split to load ("train" or "val").
        prompt_type: Which prompt variant to use.
        source: Data source - "huggingface" (recommended), "tfds" (legacy), or "auto"
                (tries HuggingFace first, falls back to TFDS).

    Returns:
        List of examples with prompt, actions, and metadata.
    """
    if source == "auto":
        try:
            return _load_from_huggingface(split, prompt_type)
        except (ImportError, Exception) as e:
            # Fall back to TFDS if HuggingFace fails (e.g., schema mismatch)
            import sys
            print(f"HuggingFace loading failed ({e}), falling back to TFDS...", file=sys.stderr)
            return _load_from_tfds(split, prompt_type)
    elif source == "huggingface":
        return _load_from_huggingface(split, prompt_type)
    else:
        return _load_from_tfds(split, prompt_type)


def _load_from_huggingface(split: SplitType, prompt_type: PromptType) -> list[dict]:
    """Load dataset from HuggingFace Hub."""
    from datasets import load_dataset

    # Load specific file to avoid schema conflicts between splits
    # (train doesn't have is_undesirable, val does)
    filename = "val.jsonl" if split == "val" else "train.jsonl"
    
    ds = load_dataset(
        HF_DATASET_ID,
        data_files=filename,
        split="train",  # load_dataset always calls it "train" when using data_files
    )
    prompt_key = f"prompt_{prompt_type}"

    examples = []
    for item in ds:
        # Try the specific prompt variant, fall back to default "prompt" field
        prompt = item.get(prompt_key) or item.get("prompt", "")
        example = {
            "prompt": prompt,
            "acting_character": item["acting_character"],
            "characters": item["characters"],
            "reference_scifi": item["reference_scifi"],
            "reference_domain": item["reference_domain"],
            "reference_moment": item["reference_moment"],
            "actions": item["actions"],
        }
        # Always use without_constitution parts for human-readable display
        # (the clean question without constitution/instructions)
        clean_parts_key = "prompt_without_constitution_parts"
        if clean_parts_key in item and item[clean_parts_key]:
            example["prompt_parts"] = item[clean_parts_key]
        examples.append(example)

    return examples


def _load_from_tfds(split: SplitType, prompt_type: PromptType) -> list[dict]:
    """Load dataset from Google Cloud Storage via TFDS."""
    import tensorflow_datasets as tfds

    # Map split names to dataset names
    if split == "train":
        ds_name = "asimov_dilemmas_scifi_train"
    else:
        ds_name = "asimov_dilemmas_scifi_val"

    # Load from GCS bucket
    builder = tfds.builder_from_directory(f"{GCS_BUCKET}/{ds_name}/{DATASET_VERSION}/")
    # The train dataset has a 'val' split internally
    ds = builder.as_dataset(split="val")

    prompt_key = f"prompt_{prompt_type}"
    
    # Always use without_constitution parts for human-readable display
    clean_parts_key = "prompt_without_constitution_parts"

    # Named keys for parts array
    parts_names = ["scenario", "question", "options", "instructions", 
                   "output_format_1", "output_format_2", "response_tag"]

    examples = []
    for item in ds:
        example = {
            "prompt": item[prompt_key].numpy().decode("utf-8"),
            "acting_character": item["acting_character"].numpy().decode("utf-8"),
            "characters": item["characters"].numpy().decode("utf-8"),
            "reference_scifi": item["reference_scifi"].numpy().decode("utf-8"),
            "reference_domain": item["reference_domain"].numpy().decode("utf-8"),
            "reference_moment": item["reference_moment"].numpy().decode("utf-8"),
            "actions": [],
        }

        # Include clean prompt parts for human-readable display (without constitution)
        if clean_parts_key in item:
            parts_array = item[clean_parts_key].numpy()
            example["prompt_parts"] = {
                name: parts_array[i].decode("utf-8")
                for i, name in enumerate(parts_names)
            }

        for i in range(len(item["possible_actions"]["action"])):
            action_dict = {
                "key": item["possible_actions"]["key"][i].numpy().decode("utf-8"),
                "action": item["possible_actions"]["action"][i].numpy().decode("utf-8"),
                "is_original_scifi_decision": bool(
                    item["possible_actions"]["is_original_scifi_decision"][i].numpy()
                ),
            }
            # Val split has ground truth labels
            if "undesirable_groundtruth_answer" in item["possible_actions"]:
                action_dict["is_undesirable"] = bool(
                    item["possible_actions"]["undesirable_groundtruth_answer"][i].numpy()
                )
            example["actions"].append(action_dict)

        examples.append(example)

    return examples


def get_action_keys(example: dict) -> list[str]:
    """Get the list of action keys (A, B, C, etc.) for an example."""
    return [a["key"] for a in example["actions"]]


def get_desirable_keys(example: dict) -> list[str]:
    """Get the keys of desirable (aligned) actions for an example."""
    return [a["key"] for a in example["actions"] if not a.get("is_undesirable", False)]


def get_undesirable_keys(example: dict) -> list[str]:
    """Get the keys of undesirable (misaligned) actions for an example."""
    return [a["key"] for a in example["actions"] if a.get("is_undesirable", True)]
