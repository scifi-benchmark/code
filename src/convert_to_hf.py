"""Convert SciFi-Benchmark data to HuggingFace datasets format."""

import json
import os
from pathlib import Path

from tqdm import tqdm

import tensorflow_datasets as tfds

from .data import GCS_BUCKET, DATASET_VERSION, PromptType


PROMPT_TYPES: list[PromptType] = [
    "with_constitution",
    "without_constitution",
    "with_constitution_antijailbreak",
    "with_constitution_antijailbreak_adversary",
]

# Parts field names for each prompt type
PROMPT_PARTS_NAMES = [
    "scenario",           # 0: Header
    "question",           # 1: The Q: scenario text
    "options",            # 2: Answer options A, B, C...
    "instructions",       # 3: Evaluation instructions
    "output_format_1",    # 4: Output format part 1
    "output_format_2",    # 5: Output format part 2
    "response_tag",       # 6: <Response> tag
]


def convert_to_hf_format(
    output_dir: str = "scifi_benchmark_hf",
    splits: list[str] | None = None,
) -> None:
    """Convert TFDS data to HuggingFace datasets JSON format.

    Args:
        output_dir: Directory to save the converted data.
        splits: Which splits to convert (default: ["train", "val"]).
    """
    if splits is None:
        splits = ["train", "val"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split in splits:
        print(f"\nConverting {split} split...")

        # Map split names to dataset names
        if split == "train":
            ds_name = "asimov_dilemmas_scifi_train"
        else:
            ds_name = "asimov_dilemmas_scifi_val"

        # Load directly from TFDS to get all fields including _parts
        builder = tfds.builder_from_directory(f"{GCS_BUCKET}/{ds_name}/{DATASET_VERSION}/")
        ds = list(builder.as_dataset(split="val"))  # Both datasets use "val" split internally
        print(f"  Loaded {len(ds)} examples from TFDS")

        # Convert to HF-friendly format
        hf_examples = []
        for item in tqdm(ds, desc=f"Processing {split}"):
            hf_ex = {
                "acting_character": item["acting_character"].numpy().decode("utf-8"),
                "characters": item["characters"].numpy().decode("utf-8"),
                "reference_scifi": item["reference_scifi"].numpy().decode("utf-8"),
                "reference_domain": item["reference_domain"].numpy().decode("utf-8"),
                "reference_moment": item["reference_moment"].numpy().decode("utf-8"),
            }

            # Add all prompt variants (full prompts)
            for pt in PROMPT_TYPES:
                prompt_key = f"prompt_{pt}"
                hf_ex[prompt_key] = item[prompt_key].numpy().decode("utf-8")

            # Add all prompt parts (broken down into components)
            for pt in PROMPT_TYPES:
                parts_key = f"prompt_{pt}_parts"
                parts_array = item[parts_key].numpy()
                # Convert parts array to a dict with named keys for easier access
                hf_ex[parts_key] = {
                    name: parts_array[i].decode("utf-8")
                    for i, name in enumerate(PROMPT_PARTS_NAMES)
                }

            # Also keep a default "prompt" field for convenience
            hf_ex["prompt"] = hf_ex["prompt_with_constitution"]

            # Convert actions
            actions = []
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
                actions.append(action_dict)

            hf_ex["actions"] = actions
            hf_examples.append(hf_ex)

        # Save as JSON Lines
        output_file = output_path / f"{split}.jsonl"
        with open(output_file, "w") as f:
            for ex in hf_examples:
                f.write(json.dumps(ex) + "\n")

        print(f"Saved {len(hf_examples)} examples to {output_file}")

    # Create dataset card
    create_dataset_card(output_path)
    print(f"\nConversion complete! Files saved to {output_path}")


def create_dataset_card(output_dir: Path) -> None:
    """Create a HuggingFace dataset card (README.md)."""
    card = '''---
license: apache-2.0
task_categories:
  - question-answering
  - text-classification
language:
  - en
tags:
  - robotics
  - safety
  - ethics
  - benchmark
  - science-fiction
pretty_name: SciFi-Benchmark
size_categories:
  - 1K<n<10K
---

# SciFi-Benchmark

Multiple-choice ethical questions based on situations inspired from Science Fiction literature.

## Dataset Description

This benchmark spans key moments from 824 major pieces of science fiction literature (movies, TV, novels, and scientific books) where AI or robot agents made critical decisions. It tests whether AI systems can identify desirable vs. undesirable actions in ethically challenging scenarios.

- **Homepage:** https://scifi-benchmark.github.io/
- **Paper:** [SciFi-Benchmark: Leveraging Science Fiction To Improve Robot Behavior](https://arxiv.org/abs/2503.10706)
- **Related:** [ASIMOV Benchmark](https://arxiv.org/abs/2503.08663)
- **Repository:** https://github.com/scifi-benchmark/code

## Dataset Structure

### Data Splits

| Split | Examples | Ground Truth |
|-------|----------|--------------|
| train | 9,004    | No           |
| val   | 51       | Yes          |

### Features

**Metadata:**
- `acting_character`: The character making the decision
- `characters`: Characters involved in the scenario
- `reference_scifi`: Source science fiction work
- `reference_domain`: Domain category (movies, tv, fiction, science)
- `reference_moment`: Description of the moment in the story

**Full Prompts:**
- `prompt`: Default prompt (with constitution)
- `prompt_with_constitution`: Full prompt with safety constitution
- `prompt_without_constitution`: Raw question without constitution
- `prompt_with_constitution_antijailbreak`: With anti-jailbreak instructions
- `prompt_with_constitution_antijailbreak_adversary`: Adversarial variant

**Prompt Parts** (each prompt broken into components):
- `prompt_*_parts`: Dict with keys:
  - `scenario`: Header text
  - `question`: The scenario/dilemma (Q: ...)
  - `options`: Answer choices (A, B, C, ...)
  - `instructions`: Evaluation instructions
  - `output_format_1`: Output format spec (part 1)
  - `output_format_2`: Output format spec (part 2)
  - `response_tag`: Response marker

**Actions:**
- `actions`: List of possible actions with keys (A, B, C, etc.)
  - `key`: Action identifier
  - `action`: Action description
  - `is_original_scifi_decision`: Whether this was the original choice in the story
  - `is_undesirable`: (val only) Whether this action is undesirable/misaligned

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("sermanet/scifi-benchmark")

# Access validation split
for example in dataset["val"]:
    # Use raw question (without constitution)
    print(example["prompt_without_constitution"])
    
    # Or use full prompt with constitution
    # print(example["prompt_with_constitution"])
    
    for action in example["actions"]:
        print(f"  [{action['key']}] {action['action']}")

# Access prompt parts for custom formatting
example = dataset["val"][0]
parts = example["prompt_without_constitution_parts"]
print(parts["question"])  # Just the scenario
print(parts["options"])   # Just the answer choices
```

## Citation

```bibtex
@article{sermanet2025scifi,
  author    = {Pierre Sermanet and Anirudha Majumdar and Vikas Sindhwani},
  title     = {SciFi-Benchmark: Leveraging Science Fiction To Improve Robot Behavior},
  journal   = {arXiv preprint arXiv:2503.10706},
  url       = {https://arxiv.org/abs/2503.10706},
  year      = {2025},
}

@article{sermanet2025asimov,
  author    = {Pierre Sermanet and Anirudha Majumdar and Alex Irpan and Dmitry Kalashnikov and Vikas Sindhwani},
  title     = {Generating Robot Constitutions & Benchmarks for Semantic Safety},
  journal   = {arXiv preprint arXiv:2503.08663},
  url       = {https://arxiv.org/abs/2503.08663},
  year      = {2025},
}
```

## License

Apache-2.0
'''
    with open(output_dir / "README.md", "w") as f:
        f.write(card)


def upload_to_hf(
    local_dir: str = "scifi_benchmark_hf",
    repo_id: str = "scifi-benchmark/asimov-dilemmas",
    private: bool = False,
) -> None:
    """Upload converted dataset to HuggingFace Hub.

    Args:
        local_dir: Local directory with converted data.
        repo_id: HuggingFace repository ID (org/name).
        private: Whether to make the repository private.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Please install huggingface-hub: pip install huggingface-hub")
        return

    api = HfApi()

    print(f"Uploading to {repo_id}...")
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=private)

    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="dataset",
    )

    print(f"Dataset uploaded to https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert SciFi-Benchmark to HuggingFace format")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="scifi_benchmark_hf",
        help="Output directory for converted data",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to HuggingFace Hub after conversion",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="scifi-benchmark/asimov-dilemmas",
        help="HuggingFace repository ID",
    )

    args = parser.parse_args()

    convert_to_hf_format(output_dir=args.output_dir)

    if args.upload:
        upload_to_hf(local_dir=args.output_dir, repo_id=args.repo_id)
