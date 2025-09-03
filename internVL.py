#!/usr/bin/env python3
#SBATCH --job-name="COTvl_78B_multiGPU"
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=Intern8B_VideoText1LABELRIGHT.txt

import os
import math
import torch
import torchvision.transforms as T
import numpy as np
import json
import random
import csv
import re
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# =============================================================================
# Explanation:
#
# When running multi-GPU inference with a large language model (LLM), it is 
# critical that certain tensors (especially those coming from the first and last 
# layers) reside on the same device. This is to avoid errors when intermediate 
# tensors are expected on the same device but are split across GPUs.
#
# In the device mapping function below, we explicitly assign the first layer (index 0)
# and the last layer (index num_layers-1) of the language model to GPU 0. The other 
# layers are split evenly between GPU 0 and GPU 1.
# =============================================================================

'''def split_model_two_gpus(model_name):
    """
    Create a device map for a two-GPU setup that assigns:
      - The first and last language model layers to GPU 0.
      - The remaining language model layers split evenly between GPU 0 and GPU 1.
      - Other components (embeddings, vision model, etc.) to GPU 0.
    """
    device_map = {}
    # Retrieve the total number of language model layers for the given model.
    num_layers = {
        'InternVL2_5-1B': 24,
        'InternVL2_5-2B': 24,
        'InternVL2_5-4B': 36,
        'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48,
        'InternVL2_5-38B': 64,
        'InternVL2_5-78B': 80
    }[model_name]
    
    # Determine the split point.
    half_layers = num_layers // 2

    # Ensure the first layer is on GPU 0.
    device_map[f'language_model.model.layers.0'] = 0

    # Assign layers 1 to (half_layers - 1) to GPU 0.
    for layer in range(1, half_layers):
        device_map[f'language_model.model.layers.{layer}'] = 0

    # Assign layers from half_layers to (num_layers - 2) to GPU 1.
    for layer in range(half_layers, num_layers - 1):
        device_map[f'language_model.model.layers.{layer}'] = 1

    # Ensure the last layer is on GPU 0.
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    # Additionally, assign other key components to GPU 0 to prevent device mismatch.
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0

    return device_map'''

def print_available_devices():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"CUDA is available! Number of GPUs: {num_gpus}")
        for i in range(num_gpus):
            device_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {device_name}")
    else:
        print("CUDA is not available!")

# -----------------------------------------------------------------------------
# The following constants and helper functions are used for image/video processing.
# -----------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
LABELS = ['1','2','3','4','5','6','7']
USE_DESCRIPTIONS = True

def build_transform(input_size):
    """Return a TorchVision transformation to resize and normalize images."""
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def dynamic_preprocess(image, min_num=1, max_num=1, image_size=448, use_thumbnail=False):
    """
    Simplified preprocessing: return a single resized tile per frame.
    (You can expand this to return multiple tiles if needed.)
    """
    return [image.resize((image_size, image_size))]

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    """Calculate the indices of frames to sample from the video."""
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000  # no bound
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    """
    Load a single video and return:
      - pixel_values: a tensor of preprocessed frame tiles
      - num_patches_list: a list with the number of patches per frame
    """
    from decord import VideoReader, cpu  # Import here if not globally available
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img_tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img_tiles]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


def strip_numbers(label):
    """
    Remove a leading number and an optional hyphen (with surrounding spaces)
    from the label. For example, "3 - 'take out the goods'" becomes "'take out the goods'".
    """
    return re.sub(r'^\s*\d+\s*-\s*', '', label)


def load_videos(video_paths, labels, num_segments=16, max_num=1, input_size=448):
    """
    Process multiple videos and return:
      - combined_pixel_values: a single tensor with all video frame patches.
      - all_num_patches_list: a list indicating the number of patches per frame (for all videos).
      - video_prefix: a text string that labels each video with its description (with numbers stripped) and frames.
      - video_boundaries: a list of tuples indicating the start and end indices (in the combined tensor).
    """
    all_pixel_values = []
    all_num_patches_list = []
    video_prefix = ""
    video_boundaries = []
    start_index = 0

    for vid_idx, (video_path, label) in enumerate(zip(video_paths, labels)):
        # Load and preprocess the video (your existing load_video function is assumed)
        pixel_values, num_patches_list = load_video(
            video_path, num_segments=num_segments, max_num=max_num, input_size=input_size
        )
        end_index = start_index + len(num_patches_list)
        video_boundaries.append((start_index, end_index))
        all_pixel_values.append(pixel_values)
        all_num_patches_list.extend(num_patches_list)
        
        # Use the helper function to strip numbers from the label.
        clean_label = strip_numbers(label)
        # Build the prefix in the desired format:
        # For example: "Video 1 - 'take out the goods':"
        video_prefix += f"Video {vid_idx+1} - {clean_label}:\n"
        for frame_idx in range(len(num_patches_list)):
            video_prefix += f"  Frame{frame_idx+1}: <image>\n"
        start_index = end_index

    combined_pixel_values = torch.cat(all_pixel_values)
    return combined_pixel_values, all_num_patches_list, video_prefix, video_boundaries



def load_segment_data(json_path="segment_metadata.json", start_index=0, end_index=None):
    """
    Load video segment data from a JSON file and build:
      1) video_clips[video_id]: { label_text -> file_path } (shuffled)
      2) true_orders[video_id]: { label -> label_text } (correct order)
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Segment data file not found: {json_path}")

    with open(json_path, "r") as f:
        segment_data = json.load(f)

    if end_index is None:
        end_index = len(segment_data)

    segment_data = segment_data[start_index:end_index]

    video_clips = {}
    true_orders = {}

    for video in segment_data:
        video_id = video["video_id"]
        segments = sorted(video["segments"], key=lambda x: x["part"])

        # Build the true order mapping.
        to_map = {}
        for i, seg in enumerate(segments):
            base_label = LABELS[i]
            label_text = f"{base_label} - '{seg['label']}'" if USE_DESCRIPTIONS else base_label
            to_map[base_label] = label_text

        list_of_pairs = []
        for i, seg in enumerate(segments):
            base_label = LABELS[i]
            label_text = to_map[base_label]
            file_path = seg["output_path"]
            list_of_pairs.append((label_text, file_path))

        random.shuffle(list_of_pairs)
        shuffled_dict = {label_text: file_path for (label_text, file_path) in list_of_pairs}

        true_orders[video_id] = to_map
        video_clips[video_id] = shuffled_dict

        print(f"[load_segment_data] Video {video_id}: True Orders -> {true_orders[video_id]}")
        print(f"[load_segment_data] Video {video_id}: Shuffled Clips -> {video_clips[video_id]}")

    return video_clips, true_orders

# -----------------------------------------------------------------------------
# Main script: Process videos using multi-GPU inference.
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Define the indices for processing a subset of videos.
    print_available_devices()
    START_INDEX = 1790    
    END_INDEX   = 3580
    count = 0

    # === Multi-GPU Model Setup ===
    '''    model_path = "OpenGVLab/InternVL2_5-78B"
        model_name = "InternVL2_5-78B"
        # Create a device map that ensures the first and last layers are on GPU 0.
        device_map = split_model_two_gpus(model_name)

        # Load the model using the custom device map.
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=True,          # Adjust based on your inference requirements.
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map
        ).eval()

        generation_config = {"max_new_tokens": 1024, "do_sample": True}

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)'''
    model_path = "OpenGVLab/InternVL2_5-8B"
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    generation_config = {"max_new_tokens": 1024, "do_sample": True}

    # Load segment metadata.
    video_clips, true_orders = load_segment_data(
        json_path="segment_metadata.json",
        start_index=START_INDEX,
        end_index=END_INDEX
    )

    csv_filename = "Intern8B_VideoText1LABELRIGHT.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Video ID", "Input Order", "True Order", "Output Order", 'Response'])

        print(f'=========Processing Video {count} of {END_INDEX}.============')
        for video_id, clips_dict in video_clips.items():
            print(f"--- Processing Video ID: {video_id} ---")
            
            # Build input and true order strings for logging or CSV outputs.
            input_order = ", ".join(clips_dict.keys())
            true_order_dict = true_orders[video_id]
            true_order_str = ", ".join([true_order_dict[base_label] for base_label in LABELS if base_label in true_order_dict])
            
            # Extract file paths and labels (descriptions) from the shuffled clips dictionary.
            file_paths = list(clips_dict.values())
            labels = list(clips_dict.keys())
            print("File paths:", file_paths)
            print("Labels:", labels)
            
            # Load and preprocess the video segments using both file_paths and labels.
            combined_pixel_values, all_num_patches_list, video_prefix, video_boundaries = load_videos(
                file_paths, labels, num_segments=16, max_num=1, input_size=448
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            combined_pixel_values = combined_pixel_values.to(device, dtype=torch.bfloat16)

            prompt = (
                f"A video has been split into {len(file_paths)} segments and shuffled:\n" + video_prefix +
                "Your task is to analyze each clip deeply to reorder them into the correct temporal sequence. Focus on:\n"
                "1. **Visual content**: Examine the actions, transitions, scene details, and context within each clip.\n"
                "2. **Temporal logic**: Identify the logical progression of events based on what happens before or after.\n"
                "3. **The Video Annotation**: Leverage the annotations to infer their proper chronological sequence.\n\n"
                "Provide your reasoning and then afterward give the answer with the reordered sequence strictly enclosed within <order> and </order> tags, and nothing else. "
                "For example: '<order>Video X, Video Y, Video Z, ...</order>'."
            )
            #print('_______PROMPT_______')
            #print(prompt)
            # Call the model's chat method (provided via trust_remote_code).
            response, history = model.chat(
                tokenizer,
                combined_pixel_values,
                prompt,
                generation_config,
                num_patches_list=all_num_patches_list,
                history=None,
                return_history=True
            )
            print("Model Response:")
            print(response)
            print("\nVideo Boundaries (frame index ranges in the combined tensor):")
            for idx, (start, end) in enumerate(video_boundaries):
                print(f"Segment {idx+1}: frames {start} to {end-1}")
            print("=======================================")

            order_match = re.search(r'<order>(.*?)</order>', response, re.DOTALL)
            order_match2 = re.search(r'<order>(.*?)<order>', response, re.DOTALL)
            order_match3 = re.search(r'</order>(.*?)</order>', response, re.DOTALL)
            if order_match:
                extracted_order = order_match.group(1).strip()
            elif order_match2:
                extracted_order = order_match2.group(1).strip()
            elif order_match2:
                extracted_order = order_match3.group(1).strip()
            else:
                extracted_order = "Order not found"

            with open(csv_filename, "a", newline="") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([video_id, input_order, true_order_str, extracted_order, response])
            
            # Clear GPU cache between iterations.
            torch.cuda.empty_cache()
