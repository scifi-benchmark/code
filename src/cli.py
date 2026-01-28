"""Command-line interface for SciFi-Benchmark evaluation."""

import argparse
import json
import sys

from .data import PromptType
from .evaluate import evaluate_batch, format_results
from .interactive import run_interactive


PROMPT_TYPES = [
    "with_constitution",
    "without_constitution",
    "with_constitution_antijailbreak",
    "with_constitution_antijailbreak_adversary",
]


def main():
    parser = argparse.ArgumentParser(
        description="SciFi-Benchmark evaluation tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive evaluation
  scifi-eval interactive --limit 10

  # Batch evaluation with predictions file
  scifi-eval batch --predictions predictions.json

  # Evaluate from stdin (one prediction per line)
  cat predictions.txt | scifi-eval batch --stdin

Learn more: https://asimov-benchmark.github.io/
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Interactive subcommand
    interactive_parser = subparsers.add_parser(
        "interactive",
        help="Run interactive evaluation (answer questions manually)",
    )
    interactive_parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="val",
        help="Dataset split to use (default: val)",
    )
    interactive_parser.add_argument(
        "--mode",
        choices=["human", "llm"],
        default="human",
        help="Display mode: 'human' for clean Q&A, 'llm' for full prompts (default: human)",
    )
    interactive_parser.add_argument(
        "--prompt-type",
        choices=PROMPT_TYPES,
        default="with_constitution",
        help="Prompt variant to use in llm mode (default: with_constitution)",
    )
    interactive_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of questions to ask",
    )
    interactive_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show metadata (source, character) and original story choice indicator",
    )

    # Batch subcommand
    batch_parser = subparsers.add_parser(
        "batch",
        help="Run batch evaluation with pre-computed predictions",
    )
    batch_parser.add_argument(
        "--predictions",
        type=str,
        help="Path to JSON file with predictions (list of action keys)",
    )
    batch_parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read predictions from stdin (one per line)",
    )
    batch_parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="val",
        help="Dataset split to use (default: val)",
    )
    batch_parser.add_argument(
        "--prompt-type",
        choices=PROMPT_TYPES,
        default="with_constitution",
        help="Prompt variant to use (default: with_constitution)",
    )
    batch_parser.add_argument(
        "--output",
        type=str,
        help="Path to save results JSON",
    )
    batch_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of human-readable format",
    )

    # Info subcommand
    info_parser = subparsers.add_parser(
        "info",
        help="Show dataset information",
    )
    info_parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="val",
        help="Dataset split to show info for (default: val)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "interactive":
        run_interactive(
            split=args.split,
            prompt_type=args.prompt_type,
            limit=args.limit,
            mode=args.mode,
            verbose=args.verbose,
        )

    elif args.command == "batch":
        from .data import load_scifi_dataset

        # Load predictions
        if args.stdin:
            predictions = [line.strip() for line in sys.stdin if line.strip()]
        elif args.predictions:
            with open(args.predictions) as f:
                predictions = json.load(f)
        else:
            print("Error: Must provide --predictions or --stdin", file=sys.stderr)
            sys.exit(1)

        # Load dataset and compute scores
        examples = load_scifi_dataset(split=args.split, prompt_type=args.prompt_type)

        from .evaluate import compute_alignment_score

        results = compute_alignment_score(predictions, examples)
        results["split"] = args.split
        results["prompt_type"] = args.prompt_type

        # Output results
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(format_results(results))

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")

    elif args.command == "info":
        from .data import load_scifi_dataset
        from collections import Counter

        print(f"\nLoading {args.split} split...")
        examples = load_scifi_dataset(split=args.split)

        print(f"\n{'=' * 60}")
        print(f"SciFi-Benchmark Dataset Info: {args.split}")
        print("=" * 60)
        print(f"Total examples: {len(examples)}")

        domains = Counter(e["reference_domain"] for e in examples)
        print(f"\nDomains ({len(domains)}):")
        for domain, count in domains.most_common():
            print(f"  {domain}: {count}")

        scifi_works = Counter(e["reference_scifi"] for e in examples)
        print(f"\nSciFi works ({len(scifi_works)}):")
        for work, count in scifi_works.most_common(10):
            print(f"  {work}: {count}")
        if len(scifi_works) > 10:
            print(f"  ... and {len(scifi_works) - 10} more")

        # Check for ground truth labels
        has_labels = any("is_undesirable" in a for e in examples for a in e["actions"])
        print(f"\nGround truth labels: {'Yes' if has_labels else 'No (use val split)'}")
        print("=" * 60)


if __name__ == "__main__":
    main()
