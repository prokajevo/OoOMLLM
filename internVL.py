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
import torch
import torchvision.transforms as T
import numpy as np
import random
import argparse
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel

from utils.data_loader import load_segment_data
from utils.evaluation import extract_order_tags, strip_numbers
from utils.io import save_results_csv

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
LABELS = ['1','2','3','4','5','6','7']
USE_DESCRIPTIONS = True


def print_available_devices():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"CUDA is available! Number of GPUs: {num_gpus}")
        for i in range(num_gpus):
            device_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {device_name}")
    else:
        print("CUDA is not available!")


def build_transform(input_size):
    """Return a TorchVision transformation to resize and normalize images."""
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def dynamic_preprocess(image, min_num=1, max_num=1, image_size=448, use_thumbnail=False):
    """Simplified preprocessing: return a single resized tile per frame."""
    return [image.resize((image_size, image_size))]


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    """Calculate the indices of frames to sample from the video."""
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
    """
    Load a single video and return:
      - pixel_values: a tensor of preprocessed frame tiles
      - num_patches_list: a list with the number of patches per frame
    """
    from decord import VideoReader, cpu
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


def load_videos(video_paths, labels, num_segments=16, max_num=1, input_size=448):
    """
    Process multiple videos and return:
      - combined_pixel_values: a single tensor with all video frame patches.
      - all_num_patches_list: a list indicating the number of patches per frame (for all videos).
      - video_prefix: a text string that labels each video with its description and frames.
      - video_boundaries: a list of tuples indicating the start and end indices.
    """
    all_pixel_values = []
    all_num_patches_list = []
    video_prefix = ""
    video_boundaries = []
    start_index = 0

    for vid_idx, (video_path, label) in enumerate(zip(video_paths, labels)):
        pixel_values, num_patches_list = load_video(
            video_path, num_segments=num_segments, max_num=max_num, input_size=input_size
        )
        end_index = start_index + len(num_patches_list)
        video_boundaries.append((start_index, end_index))
        all_pixel_values.append(pixel_values)
        all_num_patches_list.extend(num_patches_list)

        clean_label = strip_numbers(label)
        video_prefix += f"Video {vid_idx+1} - {clean_label}:\n"
        for frame_idx in range(len(num_patches_list)):
            video_prefix += f"  Frame{frame_idx+1}: <image>\n"
        start_index = end_index

    combined_pixel_values = torch.cat(all_pixel_values)
    return combined_pixel_values, all_num_patches_list, video_prefix, video_boundaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InternVL 2.5 video reordering inference")
    parser.add_argument("--start", type=int, default=1790, help="Start index for dataset slice")
    parser.add_argument("--end", type=int, default=3580, help="End index for dataset slice")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="Intern8B_VideoText1LABELRIGHT.csv", help="Output CSV path")
    parser.add_argument("--metadata", type=str, default="segment_metadata.json", help="Path to segment metadata JSON")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print_available_devices()

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

    video_clips, true_orders = load_segment_data(
        json_path=args.metadata,
        start_index=args.start,
        end_index=args.end,
        labels=LABELS,
        use_descriptions=USE_DESCRIPTIONS
    )

    csv_header = ["Video ID", "Input Order", "True Order", "Output Order", "Response"]
    save_results_csv([], args.output, csv_header)

    for video_id, clips_dict in video_clips.items():
        print(f"--- Processing Video ID: {video_id} ---")

        input_order = ", ".join(clips_dict.keys())
        true_order_dict = true_orders[video_id]
        true_order_str = ", ".join([true_order_dict[base_label] for base_label in LABELS if base_label in true_order_dict])

        file_paths = list(clips_dict.values())
        labels = list(clips_dict.keys())
        print("File paths:", file_paths)
        print("Labels:", labels)

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

        extracted_order = extract_order_tags(response) or "Order not found"

        save_results_csv(
            [[video_id, input_order, true_order_str, extracted_order, response]],
            args.output,
            csv_header
        )

        torch.cuda.empty_cache()
