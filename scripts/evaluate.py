import argparse
import asyncio
import os

from models.llama_vision import LlamaVision
from models.base_model import BaseLLMModel
from scripts.infer import predict

# Create arguments parser
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument("--model", help="Name of the LLM", type=str, default="llama_vision")
parser.add_argument("--dataset", help="Name of the dataset", type=str)
parser.add_argument("--prompt_technique", help="Prompting technique", type=str, default="vanilla")
parser.add_argument("--extract_table", help="Option to extract table", type=str, default="False")
parser.add_argument("--num_samples", help="Number of samples using for few shots prompting", type=int, default=1)

# Parse the arguments
args = parser.parse_args()

if args.model == "llama_vision":
    model = LlamaVision()

if args.dataset == "sroie":
    dataset_path = "datasets/sroie_v1"
elif args.dataset == "receipt_vn":
    dataset_path = "datasets/receipt_vn_v1"

if args.prompt_technique == "vanilla":
    prompt_instruction_path = "prompt_instructions/vanilla/vanilla_instruction_v1.txt"
    table_instruction_path = None
    
    if args.extract_table == "True":
        table_instruction_path = "prompt_instructions/vanilla/vanilla_instruction_v2.txt"

elif args.prompt_technique == "few_shots":
    prompt_instruction_path = "prompt_instructions/Few_Shots/fewshots_instruction_v1.txt"
    table_instruction_path = None
    
    if args.extract_table == "True":
        table_instruction_path = "prompt_instructions/Few_Shots/fewshots_instruction_v2.txt"

elif args.prompt_technique == "self_consistency":
    pass

elif args.prompt_technique == "cove":
    pass

else:
    raise ValueError("Invalid prompting technique")

async def get_evaluation(
        model: BaseLLMModel, 
        dataset_path: str, 
        prompt_technique: str, 
        prompt_instruction_path: str, 
        table_instruction_path: str,
        num_samples: int
    ):
    images_list = []
    for image in os.listdir(f"{dataset_path}/images"):
        image_path = f"{dataset_path}/images/{image}"
        images_list.append(image_path)

    tasks = [
        predict(
            model=model,
            prompt_technique=prompt_technique,
            prompt_instruction_path=prompt_instruction_path,
            table_instruction_path=table_instruction_path,
            image_path=image_path,
            num_samples=num_samples
        ) for image_path in images_list
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    em_scores = 0
    similarity_scores = 0
    
    success = 0
    fail = 0

    for res in results:
        try:
            scores = res[1]
            success += 1
        except Exception as e:
            # If error
            scores = {"EM": 0, "similarity_score": 0}
            fail += 1

        em = scores["EM"]
        similarity = scores["similarity_score"]

        em_scores += em
        similarity_scores += similarity
    
    em_scores = round(float(em_scores) / len(images_list), 4)
    similarity_scores = round(float(similarity_scores) / len(images_list), 4)

    return {"EM": em_scores, "similarity_scores": similarity_scores, "extract_success": success, "extract_fail": fail}

if __name__ == "__main__":
    results = asyncio.run(
        get_evaluation(
            model=model,
            dataset_path=dataset_path,
            prompt_technique=args.prompt_technique,
            prompt_instruction_path=prompt_instruction_path,
            table_instruction_path=table_instruction_path,
            num_samples=args.num_samples
        )
    )

    print(f"---- Dataset {args.dataset} ----")
    print(f"Prompting technique: {args.prompt_technique}")
    print(f"Extract table: {args.extract_table}")
    print(f"# samples: {args.num_samples}")
    print(f"Extract sucess: {results["extract_success"]}")
    print(f"Extract fail: {results["extract_fail"]}\n")
    
    print(f"EM: {results["EM"]}")
    print(f"Similarity score: {results["similarity_scores"]}\n")