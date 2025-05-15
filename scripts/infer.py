import argparse
import json
from dotenv import load_dotenv

from models.llama_vision import LlamaVision
from prompt_techniques.vanilla_prompt.vanilla_prompt import VanillaPrompt
from helpers.image_to_url import image_to_data_url
from helpers.ground_truth_reader import read_fields_from_ground_truth
from metrics import f1_score, exact_match, similarity_score

load_dotenv()

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
    image_path = args.image_path

if args.prompt_technique == "vanilla":
    prompt_instruction_path = "prompt_instructions/vanilla/vanilla_instruction_v1.txt"

if args.extract_table:
    pass

async def predict():
    image_data = image_to_data_url(image_path=image_path)
    fields = read_fields_from_ground_truth(image_path=image_path)

    response = await VanillaPrompt(
                prompt_instruction_path=prompt_instruction_path
            ).generate_response(model=model, fields=fields, image_data=image_data)

    results = json.loads(response)

    print("Result:\n", json.dumps(results, indent=4))

if __name__ == "__main__":
    import asyncio

    asyncio.run(predict())