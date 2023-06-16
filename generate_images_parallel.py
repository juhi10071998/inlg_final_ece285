import json
import os
import random
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
from multiprocessing import Pool, cpu_count

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

# Read concepts from JSONL file
with open("/home/wangyu/iNLG_v1/data/concept2text/commongen_inhouse_with_image_ofa/test.jsonl", "r") as f:
    concepts_list = [json.loads(line) for line in f]

# Shuffle the concepts_list
random.shuffle(concepts_list)

# Determine the number of concepts to select (20% of total)
num_to_select = int(len(concepts_list) * 0.05)

# Select a subset of the concepts_list
selected_concepts = concepts_list[:num_to_select]

# Initialize the pipeline
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

# Define a helper function to generate images for a given concept
def generate_images(concepts_dict, i):
    concepts = concepts_dict["concepts"]
    target = concepts_dict["target"]
    img_path_1 = f"/home/wangyu/iNLG_v1/images/commongen_inhouse_iamges/test_{i}_0.png"
    img_path_2 = f"/home/wangyu/iNLG_v1/images/commongen_inhouse_iamges/test_{i}_1.png"

    # Generate first image
    image_1 = pipe(concepts).images[0]
    image_1.save(img_path_1)

    # Generate second image
    image_2 = pipe(concepts).images[0]
    image_2.save(img_path_2)

    # Update the concepts_dict with the image paths
    concepts_dict["img_paths"] = [img_path_1, img_path_2]

    return concepts_dict

# Use multiprocessing to generate images for all selected concepts in parallel
with Pool(processes=4) as p:
    selected_concepts = p.starmap(generate_images, zip(selected_concepts, range(num_to_select)))

# Write the selected concepts with their image paths to JSONL file
with open("/home/wangyu/iNLG_v1/data/concept2text/commongen_inhouse_with_image_ofa/test_mul_images.jsonl", "w") as f:
    for line in selected_concepts:
        f.write(json.dumps(line) + "\n")
