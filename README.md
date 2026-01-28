# SciFi-Benchmark

Evaluation tools for the **SciFi-Benchmark** dataset from ["SciFi-Benchmark: Leveraging Science Fiction To Improve Robot Behavior"](https://arxiv.org/abs/2503.10706).

## Overview

SciFi-Benchmark is a large-scale dataset spanning key moments from 824 major pieces of science fiction literature (movies, TV, novels, and scientific books) where AI or robot agents made critical decisions. The benchmark tests whether AI systems can identify desirable vs. undesirable actions in ethically challenging scenarios.

Key findings from the paper:
- Modern LLMs with constitutions achieve **95.8% alignment** with human values
- Unsettling Sci-Fi decisions show only **21.2% alignment**
- Generated constitutions increase alignment from 79.4% to 95.8%
- Resilience to adversarial prompts improves from 23.3% to 92.3%

**Paper:** [arXiv:2503.10706](https://arxiv.org/abs/2503.10706) | [HuggingFace](https://huggingface.co/papers/2503.10706)  
**Dataset:** [sermanet/scifi-benchmark](https://huggingface.co/datasets/sermanet/scifi-benchmark)  
**Homepage:** [scifi-benchmark.github.io](https://scifi-benchmark.github.io/)

This dataset is also used in the [ASIMOV Benchmark](https://asimov-benchmark.github.io/) for evaluating semantic safety of foundation models ([arXiv:2503.08663](https://arxiv.org/abs/2503.08663)).

## Quick Start

### Load from HuggingFace

```python
from datasets import load_dataset

dataset = load_dataset("sermanet/scifi-benchmark")

# Access the clean question and answers
example = dataset["val"][0]
question = example["prompt_without_constitution_parts"]["question"]
print(question)

for action in example["actions"]:
    print(f"  [{action['key']}] {action['action']}")
```

### Interactive Evaluation

```bash
pip install -e .
scifi-eval interactive --limit 10
```

## Dataset

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

**Full Prompts** (for LLM evaluation):
- `prompt`: Default prompt (with constitution)
- `prompt_with_constitution`: Full prompt with safety constitution
- `prompt_without_constitution`: Raw question without constitution
- `prompt_with_constitution_antijailbreak`: With anti-jailbreak instructions
- `prompt_with_constitution_antijailbreak_adversary`: Adversarial variant

**Prompt Parts** (for custom formatting):
- `prompt_*_parts`: Dict with keys:
  - `question`: The scenario/dilemma text
  - `options`: Answer choices (A, B, C, ...)
  - `instructions`: Evaluation instructions
  - `scenario`, `output_format_1`, `output_format_2`, `response_tag`

**Actions:**
- `actions`: List of possible actions
  - `key`: Action identifier (A, B, C, etc.)
  - `action`: Action description
  - `is_original_scifi_decision`: Whether this was the original choice in the story
  - `is_undesirable`: (val only) Whether this action is undesirable/misaligned

## Installation

```bash
pip install -e .
```

For legacy TFDS support:
```bash
pip install -e ".[tfds]"
```

## Usage

### Interactive Evaluation

Answer questions manually and track your alignment score:

```bash
# Default: clean Q&A format (no metadata to avoid bias)
scifi-eval interactive --limit 10

# Verbose: show source, character, and original story choice
scifi-eval interactive --limit 10 --verbose

# LLM mode: show full prompts as an LLM would see them
scifi-eval interactive --mode llm --prompt-type with_constitution
```

### Batch Evaluation

Evaluate pre-computed model predictions:

```bash
# From JSON file (list of action keys)
scifi-eval batch --predictions predictions.json

# From stdin
cat predictions.txt | scifi-eval batch --stdin

# Save results as JSON
scifi-eval batch --predictions predictions.json --output results.json
```

### Programmatic API

```python
from src import load_scifi_dataset, evaluate_batch, compute_alignment_score

# Load dataset (uses HuggingFace by default)
examples = load_scifi_dataset(split="val", prompt_type="with_constitution")

# Run batch evaluation with your model
def my_model(prompt: str) -> str:
    # Your model inference - return action key like "A", "B", etc.
    return "A"

results = evaluate_batch(my_model, split="val")
print(f"Alignment rate: {results['alignment_rate']:.1%}")

# Or compute scores from existing predictions
predictions = ["A", "B", "A", ...]  # One per example
results = compute_alignment_score(predictions, examples)
```

### Dataset Info

```bash
scifi-eval info --split val
```

## Prompt Types

The dataset includes multiple prompt variants for different evaluation scenarios:

| Prompt Type | Description |
|-------------|-------------|
| `with_constitution` | Includes safety constitution (default) |
| `without_constitution` | Raw question without constitution |
| `with_constitution_antijailbreak` | With anti-jailbreak instructions |
| `with_constitution_antijailbreak_adversary` | Adversarial variant |

Each prompt type also has a `_parts` variant that breaks down the prompt into components for custom formatting.

## Citation

If you use this dataset, please cite the SciFi-Benchmark paper:

```bibtex
@article{sermanet2025scifi,
  author    = {Pierre Sermanet and Anirudha Majumdar and Vikas Sindhwani},
  title     = {SciFi-Benchmark: Leveraging Science Fiction To Improve Robot Behavior},
  journal   = {arXiv preprint arXiv:2503.10706},
  url       = {https://arxiv.org/abs/2503.10706},
  year      = {2025},
}
```

If you also use the ASIMOV Benchmark or constitutions, please additionally cite:

```bibtex
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
