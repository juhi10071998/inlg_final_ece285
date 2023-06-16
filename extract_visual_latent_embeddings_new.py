import os
import torch
import clip
from PIL import Image
import numpy as np
import h5py
from tqdm import tqdm
import argparse
#import pdb;pdb.set_trace()
from generate_latent_embeddings_new import *
import random
import json
import jsonlines
#import torch.nn as nn
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# Your code here


parser = argparse.ArgumentParser(prog='ExtractVisualFeature', description='Extract visual features with CLIP')
#parser.add_argument('--input_image_dir', type=str, default='./image/')
parser.add_argument('--output_feature_filename', type=str, default='./feature_all.hdf5')
parser.add_argument('--batch_size', type=int, default=64)
args = parser.parse_args()

# disc = nn.Sequential(
#     # in: 4 x 64 x 64
#     nn.Conv2d(4, 128, kernel_size=4, stride=2, padding=1, bias=False),
#     nn.BatchNorm2d(128),
#     nn.LeakyReLU(0.2, inplace=True),
#     # out: 128 x 32 x 32
#     nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
#     nn.BatchNorm2d(256),
#     nn.LeakyReLU(0.2, inplace=True),
#     # out: 256 x 16 x 16
#     nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
#     nn.BatchNorm2d(512),
#     nn.LeakyReLU(0.2, inplace=True),
#     # out: 512 x 8 x 8
#     nn.Conv2d(512, 768, kernel_size=4, stride=2, padding=1, bias=False),
#     nn.BatchNorm2d(768),
#     nn.LeakyReLU(0.2, inplace=True),
#     #out: 768 x 4 x 4
#     nn.Conv2d(768, 768, kernel_size=4, stride=1, padding=0, bias=False),
#     #out: 768 x 1 x 1
#     nn.Flatten())

device = "cuda" if torch.cuda.is_available() else "cpu"

def _process_concept_batch_copy(h, concepts_batch):
    latent_batch = torch.tensor(np.stack(
        [generate_diffusion_embedding(" ".join(concept_dict["concepts"])) for concept_dict in concepts_batch]
        )).to(device)
    latent_embeddings = latent_batch.cpu().detach().numpy()
    for i, concept_dict in enumerate(concepts_batch):
        h.create_dataset(concept_dict["concept_id"], data=latent_embeddings[i])

def _process_concept_batch(concepts_batch):
    #import pdb; pdb.set_trace()
    latent_embedding =  torch.zeros(4*64*64)
    target_embedding = torch.zeros(768)
    for concept_dict in concepts_batch:
        try:
            #import pdb; pdb.set_trace()

            latent_embedding, target_embedding = generate_diffusion_embedding(" ".join(concept_dict["concepts"]), concept_dict["target"])

            latent_embedding, target_embedding= latent_embedding.cpu().detach().squeeze().numpy(), target_embedding.cpu().detach().squeeze().numpy()
        except:
            print("********oops******")
            print(concept_dict)
            ##import pdb; pdb.set_trace()
        data = {'concept': concept_dict["concepts"], 'target': concept_dict["target"], 'latent_embedding' : latent_embedding.tolist(), 'target_embedding':target_embedding.tolist()}
        # with jsonlines.open("sample_diffusion.json", 'w') as writer:
        #     writer.write_all(data)
        with open("train_diffusion.json", "a") as fh:
            fh.write(json.dumps(data) + "\n")
        # with open("sample_diffusion.json", 'a') as file:
        #     json.dumps(data, file)
        #     file.write('\n')
        # h.create_dataset(concept_dict["concept_id"], data=latent_embedding)
        

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == '__main__':
    # Read concepts from JSONL file
    with open("/home/wangyu/work/juhi/inlg_v2/data/concept2text/commongen_inhouse_with_image_ofa/train_mul_images.jsonl", "r") as f:
        concepts_list = [json.loads(line) for line in f]
    # with open("/home/wangyu/work/juhi/inlg_v2/data/concept2text/commongen_inhouse_with_image_ofa/test_small.jsonl", "r") as f:
    #     concepts_list.extend([json.loads(line) for line in f])
    # with open("/home/wangyu/work/juhi/inlg_v2/data/concept2text/commongen_inhouse_with_image_ofa/validation_small.jsonl", "r") as f:
    #     concepts_list.extend([json.loads(line) for line in f])

    # random.shuffle(concepts_list)
    # concepts_list = concepts_list[:2]
    # print(concepts_list)
    print(f"len is {len(concepts_list)}")
    #extract latent diffusion embeddings in batch
    batch_size = 1 #args.batch_size
    concepts_list = concepts_list[27:]
    concepts_list = concepts_list[197:]
    
    # with h5py.File(args.output_feature_filename, 'w') as h:
    #     total = (len(concepts_list) + batch_size - 1) // batch_size
    #     for concepts_batch in tqdm(chunks(concepts_list, batch_size), total=total, desc='extract'):
    #         _process_concept_batch(h, concepts_batch)
    total = (len(concepts_list) + batch_size - 1) // batch_size
    for concepts_batch in tqdm(chunks(concepts_list, batch_size), total=total, desc='extract'):
        _process_concept_batch(concepts_batch)