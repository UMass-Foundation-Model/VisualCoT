from PIL import Image
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--questions', type=str, required=True, help='path to questions')
parser.add_argument('--images', type=str, required=True, help='path to coco images')
parser.add_argument('--qfeatures', type=str, required=True, help='output features path for questions')
parser.add_argument('--ifeatures', type=str, required=True, help='output features path for images')
args = parser.parse_args()

dataset = json.load(open(args.questions))
if 'question' in dataset:
    # VQAv2
    dataset = dataset['questions']
else:
    dataset = [{'question': d['question'],
                'image_id': d['image_id']} for d in dataset]
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
model = model.cuda()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
text_embeds_list = []
image_embeds_list = []
for q in tqdm(dataset):
    imageID = q['image_id']
    file_name = args.images+str(imageID).zfill(12)+".jpg"
    image = Image.open(file_name)
    inputs = processor(text=[q['question']], images=image, return_tensors="pt", padding=True)
    inputs = {k:v.cuda() for k,v in inputs.items()}
    outputs = model(**inputs)
    text_embeds_list.append(outputs['text_embeds'].detach().cpu())
    image_embeds_list.append(outputs['image_embeds'].detach().cpu())
text_embeds_list = torch.cat(text_embeds_list, dim=0)
image_embeds_list = torch.cat(image_embeds_list, dim=0)
np.save(args.ifeatures, image_embeds_list.numpy())
np.save(args.qfeatures, text_embeds_list.numpy())
