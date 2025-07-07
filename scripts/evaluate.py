import argparse
import asyncio
import os
from tqdm import tqdm

from models.llama_vision import LlamaVision
from models.qwen_vision import QwenVision
from models.base_model import BaseLLMModel
from scripts.infer import predict

# Create arguments parser
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument("--model", help="Name of the LLM", type=str, default="llama_vision")
parser.add_argument("--dataset", help="Name of the dataset", type=str)
parser.add_argument("--prompt_technique", help="Prompting technique", type=str, default="vanilla")
parser.add_argument("--base_prompt", help="Base prompting technique for self-consistency prompt", type=str, default="vanilla")
parser.add_argument("--extract_table", help="Option to extract table", type=str, default="False")
parser.add_argument("--num_samples", help="Number of samples using for few shots prompting", type=int, default=1)
parser.add_argument("--retries", help="Number of retries for self-consistency prompting", type=int, default=3)

# Parse the arguments
args = parser.parse_args()

if args.model == "llama_vision":
    model = LlamaVision()

elif args.model == "qwen_vision":
    model = QwenVision()

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
    if args.base_prompt == "vanilla":
        prompt_instruction_path = "prompt_instructions/vanilla/vanilla_instruction_v1.txt"
        table_instruction_path = None
    
        if args.extract_table == "True":
            table_instruction_path = "prompt_instructions/vanilla/vanilla_instruction_v2.txt"
    
    elif args.base_prompt == "few_shots":
        prompt_instruction_path = "prompt_instructions/Few_Shots/fewshots_instruction_v1.txt"
        table_instruction_path = None
        
        if args.extract_table == "True":
            table_instruction_path = "prompt_instructions/Few_Shots/fewshots_instruction_v2.txt"
    
    else:
        raise ValueError("Not found baseline prompt for self-consistency prompting!")

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
        num_samples: int,
        base_prompt: str,
        retries: int
    ):
    images_list = []
    for image in os.listdir(f"{dataset_path}/images"):
        image_path = f"{dataset_path}/images/{image}"
        images_list.append(image_path)

    results = []

    for image_path in tqdm(images_list):
        res = await predict(
                model=model,
                prompt_technique=prompt_technique,
                prompt_instruction_path=prompt_instruction_path,
                table_instruction_path=table_instruction_path,
                image_path=image_path,
                num_samples=num_samples,
                base_prompt=base_prompt,
                retries=retries
            )
        
        results.append(res)

    em_scores = 0
    similarity_scores_tfidf = 0
    similarity_scores_sbert = 0
    cl_precision = 0
    cl_recall = 0
    cl_f1 = 0
    precision = 0
    recall = 0
    f1 = 0

    success = 0
    fail = 0

    for res in results:
        try:
            scores = res[1]
            success += 1
        except Exception as e:
            # If error
            scores = {"EM": 0, "similarity_score_tfidf": 0, "similarity_score_sbert": 0, "cl_precision": 0, "cl_recall": 0, "cl_f1": 0, "precision": 0, "recall": 0, "f1": 0}
            fail += 1

        em_scores += scores["EM"]
        similarity_scores_tfidf += scores["similarity_score_tfidf"]
        similarity_scores_sbert += scores["similarity_score_sbert"]
        cl_precision += scores["cl_precision"]
        cl_recall += scores["cl_recall"]
        cl_f1 += scores["cl_f1"]
        precision += scores["precision"]
        recall += scores["recall"]
        f1 += scores["f1"]

    em_scores = round(float(em_scores) / len(images_list), 4)
    similarity_scores_tfidf = round(float(similarity_scores_tfidf) / len(images_list), 4)
    similarity_scores_sbert = round(float(similarity_scores_sbert) / len(images_list), 4)
    cl_precision = round(float(cl_precision) / len(images_list), 4)
    cl_recall = round(float(cl_recall) / len(images_list), 4)
    cl_f1 = round(float(cl_f1) / len(images_list), 4)

    precision = round(float(precision) / len(images_list), 4)
    recall = round(float(recall) / len(images_list), 4)
    f1 = round(float(f1) / len(images_list), 4)

    return {
            "EM": em_scores, 
            "similarity_scores_tfidf": similarity_scores_tfidf,
            "similarity_scores_sbert": similarity_scores_sbert,
            "cl_precision": cl_precision,
            "cl_recall": cl_recall,
            "cl_f1": cl_f1,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "extract_success": success, 
            "extract_fail": fail
        }

if __name__ == "__main__":
    results = asyncio.run(
        get_evaluation(
            model=model,
            dataset_path=dataset_path,
            prompt_technique=args.prompt_technique,
            prompt_instruction_path=prompt_instruction_path,
            table_instruction_path=table_instruction_path,
            num_samples=args.num_samples,
            base_prompt=args.base_prompt,
            retries=args.retries
        )
    )

    print(f"---- Dataset {args.dataset} ----")
    print(f"Model: {args.model}")
    print(f"Prompting technique: {args.prompt_technique}")

    if args.prompt_technique == "self_consistency":
        print(f"Base prompt: {args.base_prompt}")
        print(f"Retries: {args.retries}")
    
    print(f"Extract table: {args.extract_table}")
    print(f"# samples: {args.num_samples}")
    print(f"Extract sucess: {results["extract_success"]}")
    print(f"Extract fail: {results["extract_fail"]}\n")
    
    print(f"EM: {results["EM"]}")
    print(f"Similarity score (TF-IDF): {results["similarity_scores_tfidf"]}")
    print(f"Similarity score (Sentence Transformers): {results["similarity_scores_sbert"]}")
    print(f"Character-Level Precision: {results["cl_precision"]}")
    print(f"Character-Level Recall: {results["cl_recall"]}")
    print(f"Character-Level F1: {results["cl_f1"]}")
    print(f"Precision: {results["precision"]}")
    print(f"Recall: {results["recall"]}")
    print(f"F1: {results["f1"]}\n")
