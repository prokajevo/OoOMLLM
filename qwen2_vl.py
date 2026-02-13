#!/usr/bin/env python3
#SBATCH --job-name="vo_Q4_49"
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=Q4ExperimentVideoLabel10.txt

import os
import random
import json
import string
import csv
import torch
import time
import re
import argparse
import numpy as np
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from utils.data_loader import load_metadata
from utils.evaluation import extract_video_numbers
from utils.io import save_results_csv


def randomize_segments(segments):
    """Assign arbitrary filenames and shuffle loading order."""
    randomized_segments = []
    for seg in segments:
        arbitrary_name = f"{''.join(random.choices(string.ascii_letters + string.digits, k=6))}.mp4"
        seg["arbitrary_name"] = arbitrary_name
        print(f"Assigned Arbitrary Name: {arbitrary_name} to Original: {seg['output_path']}")
        randomized_segments.append(seg)
    random.shuffle(randomized_segments)
    for x in randomized_segments:
        print(x)
    return randomized_segments


def create_message(randomized_segments, video_id):
    video_inputs = [
        {
            "type": "video",
            "video": seg["output_path"],
        }
        for seg in randomized_segments
    ]

    labels_list = "\n".join([
        f"- Video {i + 1}: {seg.get('label', 'No label provided')}"
        for i, seg in enumerate(randomized_segments)
    ])

    prompt = (
        f"The video has been split into {len(randomized_segments)} clips, shuffled randomly. Each clip is labeled as follows:\n"
        f"{labels_list}\n\n"
        "Note: The order of the labels corresponds directly to the shuffled order of the video clips.\n\n"
        "Your task is to analyze each clip deeply to reorder them into the correct temporal sequence. Focus on:\n"
        "1. **Visual content**: Examine the actions, transitions, scene details, and context within each clip.\n"
        "2. **Temporal logic**: Identify the logical progression of events based on what happens before or after.\n"
        "3. **Labels**: Leverage the labels to infer their proper chronological sequence.\n\n"
        "Provide the reordered sequence as: 'Video X - [label text], Video Y - [label text], ...'."
    )

    return [
        {
            "role": "system",
            "content": "You are a strict assistant who always follows the user's instructions precisely without making assumptions or skipping steps.",
        },
        {
            "role": "user",
            "content": video_inputs + [{"type": "text", "text": prompt}],
        }
    ]


def infer_order(randomized_segments, video_id, model, processor):
    SCALING_RESOLUTIONS = [(224, 224), (160, 160), (112, 112)]
    for resolution in SCALING_RESOLUTIONS:
        try:
            print(f"Trying resolution {resolution} for video {video_id}")

            video_inputs = [
                torch.nn.functional.interpolate(video, size=resolution)
                for video in process_vision_info(create_message(randomized_segments, video_id))[1]
            ]

            inputs = processor(
                text=[processor.apply_chat_template(
                    create_message(randomized_segments, video_id),
                    tokenize=False,
                    add_generation_prompt=True,
                    add_vision_id=True
                )],
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to("cuda")

            generated_ids = model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            print(output_text)

            return output_text[0], resolution

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                torch.cuda.empty_cache()
                print(f"Out of memory at resolution {resolution}, trying lower resolution...")
                continue
            else:
                raise e

        finally:
            del video_inputs, inputs
            torch.cuda.empty_cache()

    raise RuntimeError(f"All resolutions failed for video {video_id}. Not enough memory.")


def evaluate_predictions(prediction, ground_truth, randomized_segments):
    predicted_ids = extract_video_numbers(prediction)

    input_mapping = {f"Video {i+1}": seg["arbitrary_name"] for i, seg in enumerate(randomized_segments)}
    original_mapping = {seg["arbitrary_name"]: seg["part"] for seg in randomized_segments}

    try:
        predicted_arbitrary_names = [input_mapping[video_id] for video_id in predicted_ids]
        predicted_order = [original_mapping[name] for name in predicted_arbitrary_names]
    except KeyError as e:
        print(f"KeyError: {e}. Ensure the arbitrary name exists in the mapping.")
        raise

    correct = sum(p == gt for p, gt in zip(predicted_order, ground_truth))
    accuracy = correct / len(ground_truth) * 100

    return accuracy, predicted_order, prediction


def save_partial_results(results, output_file):
    header = ["Video ID", "Ground Truth", "Predicted Order", "Accuracy", "Reasoning", "Resolution", "Input"]
    rows = []
    for result in results:
        rows.append([
            result["video_id"],
            result["ground_truth"],
            result["predicted_order"],
            f"{result['accuracy']:.2f}%",
            result["reasoning"],
            f"{result['resolution'][0]}x{result['resolution'][1]}",
            result["input"]
        ])
    save_results_csv(rows, output_file, header)


def log_memory(stage=""):
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    print(f"{stage} - Memory Allocated: {allocated} bytes, Memory Reserved: {reserved} bytes")


def process_videos(metadata_file, video_ids, model, processor, batch_size=100, output_file="output.csv"):
    metadata = load_metadata(metadata_file)
    results = []
    processed_count = 0
    skipped_count = 0
    success_count = 0
    start_time = time.time()

    total_videos = len(video_ids)
    print(f"Total videos to process: {total_videos}")

    for video_id in video_ids:
        print(f"\nProcessing Video ID: {video_id} ({processed_count + 1}/{total_videos})")
        video_data = next((video for video in metadata if video["video_id"] == video_id), None)
        if not video_data:
            print(f"Video with ID {video_id} not found. Skipping...")
            skipped_count += 1
            continue

        segments = video_data["segments"]
        ground_truth_order = [seg["part"] for seg in segments]
        randomized_segments = randomize_segments(segments)

        try:
            print(f"Starting inference for video {video_id}...")
            prediction, resolution_used = infer_order(randomized_segments, video_id, model, processor)
            print(f"Inference completed successfully for video {video_id} at resolution {resolution_used}.")

            accuracy, predicted_order, mapped_reasoning = evaluate_predictions(
                prediction, ground_truth_order, randomized_segments
            )
            print(f"Accuracy for video {video_id}: {accuracy:.2f}%")

            results.append({
                "video_id": video_id,
                "ground_truth": ground_truth_order,
                "predicted_order": predicted_order,
                "accuracy": accuracy,
                "reasoning": mapped_reasoning,
                "input": [(item['part'], item['label']) for item in randomized_segments],
                "resolution": resolution_used
            })

            success_count += 1
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"Skipping video {video_id} due to out-of-memory errors after exhausting all resolutions.")
            else:
                print(f"RuntimeError for video {video_id}: {e}")
            skipped_count += 1
        except Exception as e:
            print(f"Error processing video {video_id}: {e}")
            skipped_count += 1
        finally:
            log_memory(f"After processing video {video_id}")
            torch.cuda.empty_cache()

        processed_count += 1

        if processed_count % batch_size == 0:
            print(f"\nSaving intermediate results after processing {processed_count} videos...")
            save_partial_results(results, output_file)
            print(f"Intermediate results saved. {len(results)} entries written to {output_file}.")
            results.clear()

    if results:
        print("\nSaving remaining results...")
        save_partial_results(results, output_file)
        print(f"Final results saved. {len(results)} entries written to {output_file}.")

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("\n=== Processing Summary ===")
    print(f"Total videos processed: {processed_count}")
    print(f"Successful: {success_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Results saved to: {output_file}")
    print(f"Total runtime: {elapsed_time:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description="Qwen2-VL video reordering inference")
    parser.add_argument("--start", type=int, default=9000, help="Start index for dataset slice")
    parser.add_argument("--end", type=int, default=9878, help="End index for dataset slice")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="Q4ExperimentVideoLabel10.csv", help="Output CSV path")
    parser.add_argument("--metadata", type=str, default="segment_metadata.json", help="Path to segment metadata JSON")
    parser.add_argument("--batch-size", type=int, default=50, help="Save interval (batch size)")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4")

    metadata = load_metadata(args.metadata)
    sliced_metadata = metadata[args.start:args.end]
    video_ids = [video["video_id"] for video in sliced_metadata]

    print(f"Processing videos {args.start} to {args.end}...")
    process_videos(args.metadata, video_ids, model, processor,
                   batch_size=args.batch_size, output_file=args.output)
    print(f"Processing complete! Results saved to {args.output}.")


if __name__ == "__main__":
    main()
