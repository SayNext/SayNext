"""
 * Copyright (c) 2025.
 * All rights reserved.
 * Code for SayNext project
"""

import os
import numpy as np
import torch
import csv
import json
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from torchvision.transforms.functional import InterpolationMode
import argparse
import random

parser = argparse.ArgumentParser(description="Process input and output directories with segment limits.")
parser.add_argument("--gpu", type=str, default="0", help="Specify the GPU device(s) to use (e.g., '0', '0,1').")
parser.add_argument("--max_segments", type=int, default=32, help="Maximum number of segments to process (default: 32).")
parser.add_argument("--indir", type=str, required=True, help="Path to the input test data.")
parser.add_argument("--outdir", type=str, required=True, help="Path to save the output.")
parser.add_argument("--expfile", type=str, required=True, help="Path to reference expression file.")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num), key=lambda x: x[0] * x[1])
    target_aspect_ratio = min(target_ratios, key=lambda r: abs(aspect_ratio - r[0] / r[1]))
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))

    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    processed_images = [
        resized_img.crop((
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        ))
        for i in range(blocks)
    ]
    if use_thumbnail and blocks != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list, frame_indices

def get_exp(index, video_file, exp_caption_json_path, sampling = True):
    with open(exp_caption_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # locate to video
    if video_file not in data:
        print(f"{video_file} not in this json file.")
        return ""
    
    groups = data[video_file]
    # search in all groups，and find one including 'index'
    for group in groups:
        if group["start_index"] <= index <= group["end_index"]:
            if sampling:
                return random.choice(group["multi_caption"])
            else:
                return group["one_caption"]

    return ""


def write_predictions_to_csv(input_csv, output_csv, exp_caption_json_path, input_video_root):
    """

    Args:
        input_csv (str): path of input .csv file, including video, question and answer
        output_csv (str): path of output .csv file, including video, question, answer and prediction
        exp_caption_json_path (str): path of expression caption file, which should be generated before by 
    """
    video_question_answer_pairs = []

    with open(input_csv, 'r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            video_question_answer_pairs.append((row['video'], row['question'], row['answer']))

    # save results
    results = []
    for video_file, question_text, answer_text in video_question_answer_pairs:
        print(f"Processing video: {video_file}")
        
        video_path = os.path.join(input_video_root, video_file)
        num_segments = args.max_segments

        pixel_values, num_patches_list, frame_indices = load_video(video_path, num_segments=num_segments, max_num=1)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()

        # video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])

        sampling = True
        special_tokens = ""
        for i in frame_indices:
            # add_token = self.process_image(image)
            add_token = get_exp(i, video_file, exp_caption_json_path, sampling)
            add_token = f"in this frame {add_token}"
            special_tokens += f"Frame{i+1}: <image>, {add_token}\n"
        
        # The prompt could be customized
        prompt = (
            "You are a powerful multimodal model / professional psychologist. "
            "Please predict the athlete's next response to the reporter's question based on their facial expressions and body language in this video. "
            f"The reporter's question is: {question_text} "
            "Please respond in the following format: He/She will say:"
        )
        question_with_prefix = special_tokens + prompt
        # print(f"question with prefix: {question_with_prefix}")
        
        response = model.chat(tokenizer, pixel_values, question_with_prefix, generation_config,
                                    num_patches_list=num_patches_list, history=None, return_history=False)
        print(f"Response: {response}")

        results.append((video_file, question_text, answer_text, response))

        # print(f"Processed video: {video_file}")

    with open(output_csv, mode='w', newline='', encoding='utf-8') as prediction_file:
        writer = csv.writer(prediction_file)
        writer.writerow(["video", "question", "answer", "prediction"]) 
        writer.writerows(results) 

    print(f"All results have saved in {output_csv}")


path = './YOUR_MODEL_PATH'
print("Initializing model...")
model = AutoModel.from_pretrained(
    path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    use_flash_attn=True,
    trust_remote_code=True
).eval().cuda()
print("Model initialized successfully. Printing model structure:")
print(model)

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)


generation_config = dict(
    # no_repeat_ngram_size=3,      
    repetition_penalty=1.2,      
    do_sample=True,            
    # temperature=0.7,           
    # top_k=50,                  
    # top_p=0.95,              
    # num_beams=5,                  
    # early_stopping=True,         
    max_new_tokens=1024,
)

# generation_config = dict(max_new_tokens=1024, do_sample=False)

input_path = args.indir
output_path = args.outdir
exp_caption_json_path = args.expfile

write_predictions_to_csv(input_path, 
                         output_path, 
                         exp_caption_json_path,
                        './SayNext-PC2K-clips/')
