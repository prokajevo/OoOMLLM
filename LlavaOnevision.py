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
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, BitsAndBytesConfig
import random, os, json, csv, re

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
# Set the tokenizer's padding_side to "left" for better batched generation.
processor.tokenizer.padding_side = "left"
SEED = 47
random.seed(SEED)

START_INDEX = 0
END_INDEX   = 10

def load_segment_data(json_path="segment_metadata.json", start_index=0, end_index=None):
    """
    Loads video segment data, creating two dictionaries:
      1) video_clips[video_id]: { label_text -> file_path } (shuffled for model input)
      2) true_orders[video_id]: { label -> label_text } (correct chronological order)
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

        # Build "true order" mapping.
        to_map = {}
        for i, seg in enumerate(segments):
            base_label = LABELS[i]  # e.g. "BULL"
            if USE_DESCRIPTIONS:
                label_text = f"{base_label} - '{seg['label']}'"
            else:
                label_text = base_label
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


def strip_numbers(label):
    return re.sub(r'^\s*\d+\s*-\s*', '', label)

LABELS = ['1','2','3','4','5','6','7']
USE_DESCRIPTIONS = True  # adjust this flag as needed

if __name__ == "__main__":
    video_clips, true_orders = load_segment_data(
        json_path="segment_metadata.json",
        start_index=START_INDEX,
        end_index=END_INDEX
    )

    csv_filename = "OOVVLABEL72B_7.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Video ID", "Input Order", "True Order", "Output Order", 'Response'])

        # Iterate over each video id. For each video, process its parts.
        for video_id, clips_dict in video_clips.items():
            print(f"--- Processing Video ID: {video_id} ---")

            input_order = ", ".join(clips_dict.keys())

            # True order: order defined in the metadata. We use the LABELS order.
            true_order_dict = true_orders[video_id]
            true_order_str = ", ".join([true_order_dict[base_label] for base_label in LABELS if base_label in true_order_dict])

            file_paths = list(clips_dict.values())
            labels = list(clips_dict.keys())
            labels = [strip_numbers(label) for label in clips_dict.keys()]
            print("Labels:", labels)
            
            print("File paths:", file_paths)
            video_clips = []
            video_sizes = []
            valid = True  # Assume video is valid initially

            for path in file_paths:
                try:
                    container = av.open(path)
                    total_frames = container.streams.video[0].frames

                    # **Skip this video_id if any clip has fewer than 16 frames**
                    if total_frames < 16:
                        print(f"Skipping entire video {video_id} because a clip has only {total_frames} frames (less than 16).")
                        valid = False
                        break  # Stop processing further clips for this video

                    indices = np.linspace(0, total_frames - 1, num=16, dtype=int)
                    clip = read_video_pyav(container, indices)
                    video_clips.append(clip)

                    # Use the first frame's size as the representative size (width, height)
                    first_frame = Image.fromarray(clip[0])
                    video_sizes.append(first_frame.size)

                except Exception as e:
                    print(f"Skipping entire video {video_id} due to error: {e}")
                    valid = False
                    break  # Stop processing further clips for this video

            # **Skip processing if the video is invalid**
            if not valid:
                print(f"Skipping video {video_id} entirely due to a clip with fewer than 16 frames.")
                continue  # Skip to the next video in the loop

            num_segments = len(file_paths)
            conversation = []

            num_videos = len(video_clips)
            video_tags = [{"type": "video"} for _ in range(num_videos)]
            '''            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": (
                          f"A video has been split into {len(file_paths)} video clips and shuffled:\n"
                          "Your task is to analyze each clip deeply as part of reordering multiple clips into the correct temporal sequence.\n"
                          "Focus on:\n"
                          "1. *Visual content*: Examine the actions, transitions, scene details, and context within the clip.\n"
                          "2. *Temporal logic*: Identify the logical progression of events.\n\n"
                          "Provide the answer with the reordered sequence strictly enclosed within <order> and </order> tags, and nothing else. "
                          "For example: '<order>Video 1, Video 5, ...</order>'.")}] + video_tags
                }
            ]
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": (
                            f"A video has been split into {len(file_paths)} segments and shuffled.\n"
                            "Each segment is labeled sequentially (e.g., 'Video 1', 'Video 2'), but this does not reflect the true order.\n"
                            "Deeply analyze *each video* before determining the correct sequence. Focus on:\n"
                            "1. *Visual content*: Actions, scene details, and transitions.\n"
                            "2. *Temporal logic*: Cause-and-effect relationships.\n\n"
                            "Refer to videos by their input labels during analysis. " 
                            "Provide the video identifier while analysing each video for distinction and then afterward give the answer with the reordered sequence strictly enclosed within <order> and </order> tags, and nothing else. "
                            "For example: '<order>Video 1, Video 5 ...</order>'.")}] + video_tags
                }
            ]'''

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
                            "💡 **Think Step-by-Step**: Explain your reasoning while analyzing each clip.\n\n"
                            "🚨 **Final Answer Format**:\n"
                            "After your reasoning, return only the reordered sequence enclosed strictly within `<order>` and `</order>` tags.\n"
                            "Example:\n"
                            "`<order>Video 1, Video 4 </order>`"
                        )}] + [{"type": "video"} for _ in range(len(file_paths))]
                }
            ]

            # Use the processor to apply the chat template.
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            #print("Formatted prompt:\n", prompt)

            inputs = processor(text=prompt, videos=video_clips, return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            output_ids = model.generate(**inputs, max_new_tokens=1050)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], output_ids)]            
            response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            print("Model response:\n", response)

            print("=======================================")

            order_match = re.search(r'<order>(.*?)</order>', response, re.DOTALL)
            order_match2 = re.search(r'<order>(.*?)<order>', response, re.DOTALL)
            order_match3 = re.search(r'</order>(.*?)</order>', response, re.DOTALL)

            if order_match:
                extracted_order = order_match.group(1).strip()
            elif order_match2:
                extracted_order = order_match2.group(1).strip()

            elif order_match3:
                extracted_order = order_match3.group(1).strip()
            else:
                extracted_order = "Order not found"

            csv_writer.writerow([video_id, input_order, true_order_str, extracted_order,response])

            torch.cuda.empty_cache()
