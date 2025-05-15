import argparse

from models.llama_vision import LlamaVision
from metrics import f1_score, exact_match, similarity_score

# Create arguments parser
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument("--model", help="Name of the LLM", type=str, default="llama_vision")
parser.add_argument("--image_path", help="Path to the image", type=str)
parser.add_argument("--prompt_technique", help="Prompting technique", type=str, default="vanilla")
parser.add_argument("--extract_table", help="Option to extract table", default=False, type=bool)

# Parse the arguments
args = parser.parse_args()

if args.model == "llama_vision":
    model = LlamaVision()

if args.image_path:
    pass

if args.prompt_technique:
    pass

if args.extract_table:
    pass