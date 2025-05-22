import os
import random

from helpers.ground_truth_reader import read_formal_and_table_from_ground_truth, read_fields_from_ground_truth, read_table_column_from_ground_truth
from helpers.image_to_url import image_to_data_url

def get_random_examples(image_path: str, num_samples: int = 1):
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
