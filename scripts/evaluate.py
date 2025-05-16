import argparse
import asyncio

from models.llama_vision import LlamaVision
from scripts.infer import predict

# Create arguments parser
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument("--model", help="Name of the LLM", type=str, default="llama_vision")
parser.add_argument("--dataset", help="Name of the dataset", type=str)
parser.add_argument("--prompt_technique", help="Prompting technique", type=str, default="vanilla")
parser.add_argument("--extract_table", help="Option to extract table", default=False, type=bool)

# Parse the arguments
args = parser.parse_args()

if args.model == "llama_vision":
    model = LlamaVision()

if args.dataset == "sroie":
    dataset_path = "datasets/sroie_v1"
elif args.dataset == "receipt_vn":
    dataset_path = "datasets/receipt_vn_v1"

if args.prompt_technique not in ["vanilla", "self_consistency", "few_shot", "cove"]:
    raise ValueError("Invalid prompting technique")

if args.extract_table:
    pass