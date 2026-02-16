#!/usr/bin/env python3
#SBATCH --job-name="OneV72BLabel"
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=OOVVLABEL72B_7.txt

import av
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import random, os, argparse

from utils.data_loader import load_segment_data
from utils.evaluation import extract_order_tags, strip_numbers
from utils.io import save_results_csv

LABELS = ['1','2','3','4','5','6','7']
USE_DESCRIPTIONS = True


def read_video_pyav(container, indices):
    """
    Decode the video with PyAV decoder.
    Args:
        container: PyAV container.
        indices: List of frame indices to decode.
    Returns:
        A NumPy array of decoded frames with shape (num_frames, height, width, 3).
    """
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaVA-OneVision video reordering inference")
    parser.add_argument("--start", type=int, default=0, help="Start index for dataset slice")
    parser.add_argument("--end", type=int, default=10, help="End index for dataset slice")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="OOVVLABEL72B_7.csv", help="Output CSV path")
    parser.add_argument("--metadata", type=str, default="segment_metadata.json", help="Path to segment metadata JSON")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        "llava-hf/llava-onevision-qwen2-72b-ov-hf",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_flash_attention_2=True,
        trust_remote_code=True,
        load_in_4bit=True,
    ).to(device="cuda", non_blocking=True)
    model.eval()

    processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-72b-ov-hf")
    processor.tokenizer.padding_side = "left"

    video_clips_data, true_orders = load_segment_data(
        json_path=args.metadata,
        start_index=args.start,
        end_index=args.end,
        labels=LABELS,
        use_descriptions=USE_DESCRIPTIONS
    )

    csv_header = ["Video ID", "Input Order", "True Order", "Output Order", "Response"]
    save_results_csv([], args.output, csv_header)

    for video_id, clips_dict in video_clips_data.items():
        print(f"--- Processing Video ID: {video_id} ---")

        input_order = ", ".join(clips_dict.keys())
        true_order_dict = true_orders[video_id]
        true_order_str = ", ".join([true_order_dict[base_label] for base_label in LABELS if base_label in true_order_dict])

        file_paths = list(clips_dict.values())
        labels = [strip_numbers(label) for label in clips_dict.keys()]
        print("Labels:", labels)
        print("File paths:", file_paths)

        video_clips = []
        video_sizes = []
        valid = True

        for path in file_paths:
            try:
                container = av.open(path)
                total_frames = container.streams.video[0].frames

                if total_frames < 16:
                    print(f"Skipping entire video {video_id} because a clip has only {total_frames} frames (less than 16).")
                    valid = False
                    break

                indices = np.linspace(0, total_frames - 1, num=16, dtype=int)
                clip = read_video_pyav(container, indices)
                video_clips.append(clip)

                first_frame = Image.fromarray(clip[0])
                video_sizes.append(first_frame.size)

            except Exception as e:
                print(f"Skipping entire video {video_id} due to error: {e}")
                valid = False
                break

        if not valid:
            print(f"Skipping video {video_id} entirely due to a clip with fewer than 16 frames.")
            continue

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        f"A video has been split into {len(file_paths)} segments and shuffled.\n"
                        "Each segment is labeled but not in the correct order. Your goal is to analyze and reconstruct the correct temporal sequence.\n\n"
                        "### **Instructions**:\n"
                        "- **Visual Analysis:** Identify key actions, scene transitions, and objects.\n"
                        "- **Cause-and-Effect:** Understand how events logically follow each other.\n"
                        "- **Ensure Accuracy:** Consider context clues like lighting, character positioning, and motion continuity.\n"
                        "- **Leverage Annotation:** Use the provided video annotation to aid in ordering when needed.\n\n"
                        "#### **Video Clips** (Input Order):\n"
                        + "\n".join([f"- **Video {i+1}**: {label}" for i, label in enumerate(labels)]) + "\n\n"
                        "Think Step-by-Step: Explain your reasoning while analyzing each clip.\n\n"
                        "Final Answer Format:\n"
                        "After your reasoning, return only the reordered sequence enclosed strictly within `<order>` and `</order>` tags.\n"
                        "Example:\n"
                        "`<order>Video 1, Video 4 </order>`"
                    )}] + [{"type": "video"} for _ in range(len(file_paths))]
            }
        ]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = processor(text=prompt, videos=video_clips, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        output_ids = model.generate(**inputs, max_new_tokens=1050)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], output_ids)]
        response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        print("Model response:\n", response)
        print("=======================================")

        extracted_order = extract_order_tags(response) or "Order not found"

        save_results_csv(
            [[video_id, input_order, true_order_str, extracted_order, response]],
            args.output,
            csv_header
        )

        torch.cuda.empty_cache()
