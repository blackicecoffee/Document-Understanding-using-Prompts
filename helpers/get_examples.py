from transformers import AutoImageProcessor, AutoModelForImageClassification
import chromadb
from PIL import Image
import torch

import os
import random

from helpers.ground_truth_reader import (
    read_formal_and_table_from_ground_truth, 
    read_fields_from_ground_truth, 
    read_table_column_from_ground_truth
)

from helpers.image_to_url import image_to_data_url

image_embedding_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
image_embedding_model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
client_eng = chromadb.PersistentClient("vectordb/eng/")
client_vn = chromadb.PersistentClient("vectordb/vn/")

def get_image_embedding(image_path: str):
    image = Image.open(image_path).convert("RGB")
    image_inputs = image_embedding_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        image_outputs = image_embedding_model(**image_inputs, output_hidden_states=True)

    # Get the [CLS] token embedding (representation of whole image)
    image_embedding = image_outputs.hidden_states[-1][:, 0, :].squeeze().numpy()

    return image_embedding

def add_image_to_vector_db(dataset_path: str, lang: str = "eng"):
    if lang == "eng":
        image_collection = client_eng.get_or_create_collection(
            name="image_embedding_eng"
        )
    else:
        image_collection = client_vn.get_or_create_collection(
            name="image_embedding_vn"
        )

    for image_name in os.listdir(dataset_path):
        image_id = image_name.split(".")[0]
        image_path = f"{dataset_path}/{image_name}"
        
        image_embedding = get_image_embedding(image_path=image_path)

        image_collection.add(
            ids=[image_id],
            embeddings=[image_embedding.tolist()],
            metadatas=[{"image_name": image_name}]
        )
    

def get_random_examples_wo_img(image_path: str, num_samples: int = 1):
    """Get random samples without image"""

    image_name = image_path.split("/")[-1]
    dataset_path = "/".join(image_path.split("/")[:-1])
    
    images_list = [f"{dataset_path}/{image}" for image in os.listdir(dataset_path) if image != image_name]
    samples_image = random.sample(images_list, k=num_samples)

    samples = []

    for image_path in samples_image:
        formal, table = read_formal_and_table_from_ground_truth(image_path=image_path)
        fields = read_fields_from_ground_truth(image_path=image_path)
        table_columns = read_table_column_from_ground_truth(image_path=image_path)

        samples.append(
            {
                "fields": fields,
                "table_columns": table_columns,
                "formal": formal,
                "table": table
            }
        )

    return samples

def get_random_examples_with_img(image_path: str, num_samples: int = 1):
    """Get random samples with image"""

    image_name = image_path.split("/")[-1]
    dataset_path = "/".join(image_path.split("/")[:-1])
    
    images_list = [f"{dataset_path}/{image}" for image in os.listdir(dataset_path) if image != image_name]
    samples_image = random.sample(images_list, k=num_samples)

    samples = []

    for image_path in samples_image:
        image_data = image_to_data_url(image_path=image_path)
        formal, table = read_formal_and_table_from_ground_truth(image_path=image_path)
        fields = read_fields_from_ground_truth(image_path=image_path)
        table_columns = read_table_column_from_ground_truth(image_path=image_path)

        samples.append(
            {
                "image_data": image_data,
                "fields": fields,
                "table_columns": table_columns,
                "formal": formal,
                "table": table
            }
        )

    return samples

def get_selected_examples_wo_img(image_path: str, num_samples: int = 1):
    """Get similar samples without image using image embedding model and similarity search"""
    pass

def get_selected_examples_with_img(image_path: str, num_samples: int = 1):
    """Get similar samples with image using image embedding model and similarity search"""
    image_name = image_path.split("/")[-1]
    dataset_path = "/".join(image_path.split("/")[:-1])

    if dataset_path.split("/")[1] == "sroie_v1":
        lang = "eng"
    else: lang = "vn"

    if lang == "eng":
        image_collection = client_eng.list_collections()
        if len(image_collection) == 0:
            add_image_to_vector_db(dataset_path=dataset_path, lang=lang)
        image_collection = client_eng.get_collection(name="image_embedding_eng")

    else:
        image_collection = client_vn.list_collections()
        if len(image_collection) == 0:
            add_image_to_vector_db(dataset_path=dataset_path, lang=lang)
        image_collection = client_vn.get_collection(name="image_embedding_vn")
    
    image_embedding = get_image_embedding(image_path=image_path)

    similar_images = image_collection.query(
        query_embeddings=[image_embedding.tolist()],
        n_results=num_samples+1
    )

    samples = []

    for idx, image_id in enumerate(similar_images["ids"][0]):
        sample_image_name = similar_images["metadatas"][0][idx]["image_name"]
        if sample_image_name != image_name:
            sample_image_path = f"{dataset_path}/{sample_image_name}"
            image_data = image_to_data_url(image_path=sample_image_path)
            formal, table = read_formal_and_table_from_ground_truth(image_path=sample_image_path)
            fields = read_fields_from_ground_truth(image_path=sample_image_path)
            table_columns = read_table_column_from_ground_truth(image_path=sample_image_path)

            samples.append(
                {
                    "image_name": sample_image_name,
                    "image_data": image_data,
                    "fields": fields,
                    "table_columns": table_columns,
                    "formal": formal,
                    "table": table
                }
            )
    
    return samples