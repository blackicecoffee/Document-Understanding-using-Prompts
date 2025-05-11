import argparse

# Create arguments parser
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument("--model", help="Name of the LLM", type=str)
parser.add_argument("--dataset", help="Name of the dataset", type=str)
parser.add_argument("--prompt_technique", help="Prompting technique", type=str, default="vanilla")
parser.add_argument("--extract_table", help="Option to extract table", default=False, type=bool)

# Parse the arguments
args = parser.parse_args()

if args.model:
    pass

if args.dataset:
    pass

if args.prompt_technique:
    pass

if args.extract_table:
    pass