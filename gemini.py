#!/usr/bin/env python3
#SBATCH --job-name="Gemini run2"
#SBATCH --time=48:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=Gemini1.5FlashVideo4_run.txt

import os
import uuid
import json
import csv
import base64
import mimetypes
import random
import re
import time
import subprocess
import argparse
from collections import deque

import google.generativeai as genai

from utils.data_loader import load_segment_data
from utils.evaluation import extract_order_tags, extract_labels_from_order, compute_accuracy
from utils.io import save_results_csv

LABELS = ["BULL", "SPADE", "HEART", "DIAMOND", "CLUB", "STAR", "MOON", "SUN"]

API_RATE_LIMIT = 10  # 10 requests per minute
REQUEST_WINDOW = 60  # 60 seconds
request_times = deque()


def remove_audio(input_video):
    """Removes audio from the video using FFmpeg."""
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Video file not found: {input_video}")

    anonymized_filename = f"{uuid.uuid4().hex}.mp4"
    output_video = os.path.join(os.path.dirname(input_video), anonymized_filename)

    command = ["ffmpeg", "-i", input_video, "-c:v", "copy", "-an", output_video, "-y"]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return output_video


def upload_to_gemini(original_path, mime_type=None, max_retries=5):
    """Uploads a video to Gemini and retries on failure."""
    temp_files = []
    attempt = 0
    while attempt < max_retries:
        try:
            silent_video = remove_audio(original_path)
            temp_files.append(silent_video)
            file = genai.upload_file(silent_video, mime_type=mime_type)
            print(f"Uploaded file: {file.uri}")
            return file
        except Exception as e:
            print(f"Upload failed (Attempt {attempt+1}/{max_retries}): {str(e)}")
            time.sleep(2 ** attempt)
            attempt += 1
        finally:
            for tf in temp_files:
                try:
                    os.remove(tf)
                except OSError:
                    pass
            temp_files.clear()

    raise Exception(f"Failed to upload file after {max_retries} attempts")


def wait_for_files_active(files):
    """Waits for uploaded files to become ACTIVE before using them."""
    print("Waiting for file processing...")
    for name in (file.name for file in files):
        file = genai.get_file(name)
        while file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(10)
            file = genai.get_file(name)
        if file.state.name != "ACTIVE":
            raise Exception(f"File {file.name} failed to process")
    print("...all files ready\n")


def encode_video(video_path):
    """Encodes a video file into base64 format."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    mime_type, _ = mimetypes.guess_type(video_path)
    with open(video_path, 'rb') as f:
        video_data = base64.b64encode(f.read()).decode('utf-8')

    return {"mime_type": mime_type, "data": video_data}


def call_gemini_with_limit(parts):
    """Dynamically ensures we do not exceed 10 requests per minute."""
    global request_times

    current_time = time.time()
    while request_times and request_times[0] < current_time - REQUEST_WINDOW:
        request_times.popleft()

    if len(request_times) >= API_RATE_LIMIT:
        time_to_wait = REQUEST_WINDOW - (current_time - request_times[0])
        print(f"Rate limit reached! Waiting {round(time_to_wait, 2)}s before next request...")
        time.sleep(time_to_wait)

    request_times.append(time.time())
    model = genai.GenerativeModel('models/gemini-1.5-flash')
    return model.generate_content(parts)


def analyze_video_order(video_id, clips):
    """
    Uploads video clips, waits for processing, and sends them to Gemini.

    Args:
        video_id (str): Unique identifier for the video.
        clips (dict): Dictionary mapping clip labels to file paths.

    Returns:
        str: The reordered sequence returned by Gemini.
    """
    try:
        uploaded_files = [upload_to_gemini(path) for _, path in clips.items()]
        wait_for_files_active(uploaded_files)

        prompt = (
            f"A video has been split into {len(clips)} clips and shuffled randomly.\n"
            "Your task is to analyze each clip deeply and reorder them into the correct temporal sequence.\n"
            "Focus on:\n"
            "1. *Visual content*: Examine the actions, transitions, scene details, and context within each clip.\n"
            "2. *Temporal logic*: Identify the logical progression of events based on what happens before or after.\n\n"
            "Provide the reordered sequence strictly within order tags in this format: '<order>Video X , Video Y, ...</order>'."
        )

        parts = [prompt]
        for label, file in zip(clips.keys(), uploaded_files):
            parts.append(f"Clip label: {label}")
            parts.append(file)

        response = call_gemini_with_limit(parts)

        if response and response.text:
            print(f"Model Response for {video_id}: {response.text}")
            return response.text
        else:
            print(f"Empty response received for {video_id}.")
            return "ERROR: Empty response"

    except Exception as e:
        print(f"Error analyzing video {video_id}: {str(e)}")
        return f"ERROR: {str(e)}"


def process_and_save(results, true_orders, output_csv):
    """Process results and save to CSV."""
    header = ["Video ID", "Predicted Order", "True Order", "Accuracy (%)", "Inference Return"]
    rows = []
    for video_id, predicted_order in results.items():
        true_order_labels = list(true_orders[video_id].keys())
        order_text = extract_order_tags(predicted_order)
        predicted_labels = extract_labels_from_order(order_text, LABELS) if order_text else []
        accuracy = compute_accuracy(predicted_labels, true_order_labels)
        print(f"Writing to CSV: {video_id}, {predicted_labels}, {true_order_labels}, {accuracy}")
        rows.append([video_id, predicted_labels, ", ".join(true_order_labels), accuracy, predicted_order])
    save_results_csv(rows, output_csv, header)


def main():
    """Main execution loop with periodic CSV saving."""
    parser = argparse.ArgumentParser(description="Gemini 1.5 Flash video reordering inference")
    parser.add_argument("--start", type=int, default=3000, help="Start index for dataset slice")
    parser.add_argument("--end", type=int, default=3580, help="End index for dataset slice")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="Gemini1.5FlashVideo4.csv", help="Output CSV path")
    parser.add_argument("--metadata", type=str, default="segment_metadata.json", help="Path to segment metadata JSON")
    parser.add_argument("--api-key", type=str, default=None, help="Gemini API key (defaults to GEMINI_API_KEY env var)")
    args = parser.parse_args()

    random.seed(args.seed)

    api_key = args.api_key or os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)

    save_interval = 10

    try:
        video_data, true_orders = load_segment_data(
            json_path=args.metadata, start_index=args.start, end_index=args.end,
            labels=LABELS, use_descriptions=False
        )
        results = {}

        for i, (video_id, clips) in enumerate(video_data.items(), 1):
            result = analyze_video_order(video_id, clips)
            print(f"Model Output for {video_id}: {result}")
            results[video_id] = result

            if i % save_interval == 0:
                print(f"Saving results at iteration {i}")
                process_and_save(results, true_orders, args.output)
                results.clear()

        if results:
            print("Final saving of remaining results.")
            process_and_save(results, true_orders, args.output)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
