import json
import argparse
import ast
from trainer import train


def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    param.update(args)  # Add parameters from json
    train(param)


def load_json(settings_path) -> dict:
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(description="Prompt2Guard - training part.")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/cddb_training.json",
        help="Json file of settings.",
    )
    parser.add_argument(
        "--K", type=int, default=argparse.SUPPRESS, help="Number of prompts."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=argparse.SUPPRESS,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--batch_size_eval",
        type=int,
        default=argparse.SUPPRESS,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--torch_seed",
        type=int,
        default=argparse.SUPPRESS,
        help="Seed for PyTorch random number generator.",
    )
    parser.add_argument(
        "--lrate", type=float, default=argparse.SUPPRESS, help="LR for task > 0."
    )
    parser.add_argument(
        "--init_lr",
        type=float,
        default=argparse.SUPPRESS,
        help="Initial LR for task 0.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=argparse.SUPPRESS,
        help="Epochs for the other tasks.",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging."
    )
    parser.add_argument(
        "--run_name", type=str, default="testToTrash", help="Run name for WandB."
    )
    parser.add_argument(
        "--warmup_epoch",
        type=int,
        default=argparse.SUPPRESS,
        help="Number of warmup epochs.",
    )
    parser.add_argument(
        "--topk_classes", type=int, default=argparse.SUPPRESS, help="TopK classes."
    )
    parser.add_argument(
        "--ensembling",
        type=ast.literal_eval,
        default=argparse.SUPPRESS,
        help="List of boolean values for ensembling.",
    )
    return parser


if __name__ == "__main__":
    main()
