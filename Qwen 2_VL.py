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
import uuid
import string
import csv
import torch
import time
from typing import Dict
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import re
import numpy as np

output = 'Q4ExperimentVideoLabel10.csv'
# Set random seed for reproducibility
'''SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)'''


model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
 )

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4")

"""
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    device_map="auto",
    attn_implementation="flash_attention_2",  
    torch_dtype=torch.float16,  
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

"""
# Load metadata
def load_metadata(json_file):
    with open(json_file, "r") as f:
        return json.load(f)


# Assign arbitrary filenames and shuffle loading order
def randomize_segments(segments):
    randomized_segments = []
    for seg in segments:
        # Generate a random 6-character alphanumeric string
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
            #"filename": seg["arbitrary_name"],
        }
        for seg in randomized_segments
    ]
    '''
    labels_list = "\n".join([
        f"- {seg.get('label', 'No label provided')}"
        for seg in randomized_segments
    ])
    
    # Prompt emphasizing random input order
    prompt = (
        f"These are parts of a video that have been randomly shuffled into {len(randomized_segments)} clips. "
        "Each clip has a label describing its content. The input order of the clips is completely random and does not reflect their correct temporal sequence. "
        "Your task is to reorder them into the correct temporal sequence based on their visual and contextual content, as well as the labels provided.\n\n"
        f"Labels for the clips:\n{labels_list}\n\n"
        "Important: Do not assume the input order reflects the correct temporal order.\n\n"
        "Criteria for reordering:\n"
        "1. **Visual content**: Analyze the actions, transitions, and scene changes in each clip.\n"
        "2. **Temporal relationships**: Consider the logical progression of events (e.g., beginning, middle, end).\n"
        "3. **Labels**: Use the labels to infer their logical order.\n\n"
        "Return the result as a list of videos and their respective labels in their correct temporal order, such as 'video 3 - label, video 1 - label, video 2 - label'."
    )'''
    '''
    prompt = (
        f"The video has been split into {len(randomized_segments)} clips, shuffled randomly, and each clip is labeled as follows:\n{labels_list}\n\n"
        "Note: The order of the labels corresponds directly to the shuffled order of the video clips.\n\n"
        "Your task is to analyze each clip deeply to reorder them into the correct temporal sequence. Focus on:\n"
        "1. **Visual content**: Examine the actions, transitions, scene details, and context within each clip.\n"
        "2. **Temporal logic**: Identify the logical progression of events based on what happens before or after.\n"
        "3. **Labels**: Leverage the labels to infer their proper chronological sequence.\n\n"
        "Provide the reordered sequence as: 'Video X - [label text], Video Y - [label text], ...'."
    )'''

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
            "content": 'You are a strict assistant who always follows the user’s instructions precisely without making assumptions or skipping steps.',
        },
        {
            "role": "user",
            "content": video_inputs + [{"type": "text", "text": prompt}],
        }
    ]

def infer_order(randomized_segments, video_id):
    SCALING_RESOLUTIONS = [(224, 224), (160, 160), (112, 112)]  
    for resolution in SCALING_RESOLUTIONS:
        try:
            print(f"Trying resolution {resolution} for video {video_id}")

            # Resize videos to the current resolution
            video_inputs = [
                torch.nn.functional.interpolate(video, size=resolution)
                for video in process_vision_info(create_message(randomized_segments, video_id))[1]
            ]

            inputs = processor(
                text=[processor.apply_chat_template(
                    create_message(randomized_segments, video_id),
                    tokenize=False,
                    add_generation_prompt=True, 
                    add_vision_id = True
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

    # If all resolutions fail, raise an error
    raise RuntimeError(f"All resolutions failed for video {video_id}. Not enough memory.")


def extract_video_ids(prediction_text):
    """
    Extract unique video IDs (e.g., Video 1, video 1, VIDEO 1) in order of appearance,
    allowing for case insensitivity and minor format variations.
    """
    # Use regex to find matches in the form '(Video|video|VIDEO) N' (e.g., 'Video 1', 'video 2')
    matches = re.findall(r'\b(?:Video|video|VIDEO) ?(\d+)\b', prediction_text, re.IGNORECASE)
    # Prepend 'Video' to ensure uniform output format and return unique matches
    return [f"Video {num}" for num in dict.fromkeys(matches)]



def evaluate_predictions(prediction, ground_truth, randomized_segments):
    # Extract predicted IDs (e.g., 'Video 1', 'Video 2') using the refined regex
    predicted_ids = extract_video_ids(prediction)

    # Map input order to arbitrary names
    input_mapping = {f"Video {i+1}": seg["arbitrary_name"] for i, seg in enumerate(randomized_segments)}

    # Map arbitrary names to original parts
    original_mapping = {seg["arbitrary_name"]: seg["part"] for seg in randomized_segments}

    try:
        # Map predicted IDs back to arbitrary names
        predicted_arbitrary_names = [input_mapping[video_id] for video_id in predicted_ids]

        # Map arbitrary names to ground truth parts
        predicted_order = [original_mapping[name] for name in predicted_arbitrary_names]
    except KeyError as e:
        print(f"KeyError: {e}. Ensure the arbitrary name exists in the mapping.")
        raise

    # accuracy calculation
    correct = sum(p == gt for p, gt in zip(predicted_order, ground_truth))
    accuracy = correct / len(ground_truth) * 100

    return accuracy, predicted_order, prediction


def save_partial_results(results, output_file=output):
    file_exists = os.path.exists(output_file)

    with open(output_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Video ID", "Ground Truth", "Predicted Order", "Accuracy", "Reasoning", "Resolution", 'Input'])
        for result in results:
            writer.writerow([
                result["video_id"],
                result["ground_truth"],
                result["predicted_order"],
                f"{result['accuracy']:.2f}%",
                result["reasoning"],
                f"{result['resolution'][0]}x{result['resolution'][1]}",
                result["input"]
            ])


def log_memory(stage=""):
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    print(f"{stage} - Memory Allocated: {allocated} bytes, Memory Reserved: {reserved} bytes")


def process_videos(metadata_file, video_ids, batch_size=100, output_file=output):
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
            prediction, resolution_used = infer_order(randomized_segments, video_id)
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
                'input': [(item['part'], item['label']) for item in randomized_segments],
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

    # Save any remaining results
    if results:
        print("\nSaving remaining results...")
        save_partial_results(results, output_file)
        print(f"Final results saved. {len(results)} entries written to {output_file}.")

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Final Summary
    print("\n=== Processing Summary ===")
    print(f"Total videos processed: {processed_count}")
    print(f"Successful: {success_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Results saved to: {output_file}")
    print(f"Total runtime: {elapsed_time:.2f} seconds")


def main(metadata_file, start_index=0, end_index=None, batch_size=100, output_file=output):
    metadata = load_metadata(metadata_file)
    
    # Apply slicing to split the dataset
    sliced_metadata = metadata[start_index:end_index]
    video_ids = [video["video_id"] for video in sliced_metadata]
    
    print(f"Processing videos {start_index} to {end_index or len(metadata)}...")
    process_videos(metadata_file, video_ids, batch_size=batch_size, output_file=output_file)
    print(f"Processing complete! Results saved to {output_file}.")


main("segment_metadata.json", start_index=9000, end_index=9878, batch_size=50, output_file=output)