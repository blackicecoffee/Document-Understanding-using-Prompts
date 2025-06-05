import argparse
import json
import asyncio
from dotenv import load_dotenv

from models.llama_vision import LlamaVision
from models.base_model import BaseLLMModel
from prompt_techniques.vanilla_prompt.vanilla_prompt import VanillaPrompt
from prompt_techniques.Few_Shots.few_shot_prompt import FewShotsPrompt
from prompt_techniques.self_consistency.self_consistency_prompt import SelfConsistencyPrompt
from helpers.image_to_url import image_to_data_url
from helpers.ground_truth_reader import read_ground_truth, read_fields_from_ground_truth, read_table_column_from_ground_truth
from helpers.get_examples import get_random_examples
from helpers.metrics.all_scores import get_all_scores

load_dotenv()

async def predict(
        model: BaseLLMModel, 
        prompt_technique: str, 
        prompt_instruction_path: str, 
        table_instruction_path: str, 
        image_path: str,
        num_samples: int,
        base_prompt: str,
        retries: int
    ):
    await asyncio.sleep(0)

    image_data = image_to_data_url(image_path=image_path)
    fields = read_fields_from_ground_truth(image_path=image_path)
    table_columns = read_table_column_from_ground_truth(image_path=image_path)
    ground_truth = read_ground_truth(image_path=image_path)

    if prompt_technique == "vanilla":
        results = await VanillaPrompt(
                    prompt_instruction_path=prompt_instruction_path,
                    table_instruction_path=table_instruction_path
                ).generate_response(model=model, fields=fields, table_columns=table_columns, image_data=image_data)
        
    elif prompt_technique == "few_shots":
        examples = get_random_examples(image_path=image_path, num_samples=num_samples)

        results = await FewShotsPrompt(
            prompt_instruction_path=prompt_instruction_path,
            table_instruction_path=table_instruction_path,
            examples=examples
        ).generate_response(model=model, fields=fields, table_columns=table_columns, image_data=image_data)

    elif prompt_technique == "self_consistency":
        examples = get_random_examples(image_path=image_path, num_samples=num_samples)
        
        results = await SelfConsistencyPrompt(
            prompt_instruction_path=prompt_instruction_path,
            table_instruction_path=table_instruction_path,
            baseline_prompt=base_prompt,
            examples=examples,
            k=retries
        ).generate_response(model=model, fields=fields, table_columns=table_columns, image_data=image_data)

    scores = get_all_scores(ground_truth=ground_truth, pred=results)

    return [results, scores]
    
if __name__ == "__main__":
    # Create arguments parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--model", help="Name of the LLM", type=str, default="llama_vision")
    parser.add_argument("--image_path", help="Path to the image", type=str)
    parser.add_argument("--prompt_technique", help="Prompting technique", type=str, default="vanilla")
    parser.add_argument("--base_prompt", help="Base prompting technique for self-consistency prompt", type=str, default="vanilla")
    parser.add_argument("--extract_table", help="Option to extract table", type=str, default="False")
    parser.add_argument("--num_samples", help="Number of samples using for few shots prompting", type=int, default=1)
    parser.add_argument("--retries", help="Number of retries for self-consistency prompting", type=int, default=3)

    # Parse the arguments
    args = parser.parse_args()

    if args.model == "llama_vision":
        model = LlamaVision()

    if args.image_path:
        if len(args.image_path) == 0: raise ValueError("No image found!")
        image_path = args.image_path

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

    results, scores = asyncio.run(
        predict(
            model=model,
            prompt_technique=args.prompt_technique,
            prompt_instruction_path=prompt_instruction_path, 
            table_instruction_path=table_instruction_path,
            image_path=image_path,
            num_samples=args.num_samples,
            base_prompt=args.base_prompt,
            retries=args.retries
        )
    )

    print(f"Prompting technique: {args.prompt_technique}")

    if args.prompt_technique == "self_consistency":
        print(f"Base prompt: {args.base_prompt}")
        print(f"Retries: {args.retries}")

    print(f"Extract table: {args.extract_table}")
    print(f"# samples: {args.num_samples}\n")
    print("Result:\n", json.dumps(results, indent=4))
    print("\nEM score: ", scores["EM"])
    print("Similarity score (TF-IDF): ", scores["similarity_score_tfidf"])
    print("Similarity score (Sentence Transformers): ", scores["similarity_score_sbert"])
    print("Precision: ", scores["precision"])
    print("Recall: ", scores["recall"])
    print("F1:", scores["f1"])