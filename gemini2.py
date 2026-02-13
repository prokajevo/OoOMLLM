#!/usr/bin/env python3
#SBATCH --job-name="Gemini"
#SBATCH --time=48:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=GeminiPaper5.txt

import os
import uuid
import time
import re
import random
import subprocess
import argparse
from collections import deque

import google.generativeai as genai

from utils.data_loader import load_segment_data
from utils.evaluation import extract_order_tags, compute_accuracy
from utils.io import save_results_csv

# -------------------------------------------------------------------
# ----------------------- CONFIGURATIONS ----------------------------
# -------------------------------------------------------------------

USE_DESCRIPTIONS = True

LABELS = ["BULL", "SPADE", "HEART", "DIAMOND", "CLUB", "STAR", "MOON", "SUN"]

API_RATE_LIMIT = 10    # 10 requests per minute
REQUEST_WINDOW = 60    # 60 seconds
request_times = deque()


# -------------------------------------------------------------------
# ------------------------- CORE FUNCTIONS --------------------------
# -------------------------------------------------------------------

def remove_audio(input_video):
    """Removes audio from the video using FFmpeg to reduce file size and anonymize audio."""
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Video file not found: {input_video}")

    anonymized_filename = f"{uuid.uuid4().hex}.mp4"
    output_video = os.path.join(os.path.dirname(input_video), anonymized_filename)

    command = [
        "ffmpeg", "-i", input_video,
        "-c:v", "copy",
        "-an",
        output_video,
        "-y"
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return output_video


def upload_to_gemini(original_path, max_retries=5):
    """
    Uploads a video to Gemini with a unique anonymized filename.
    Uses exponential backoff and checks file size before upload.
    """
    if not os.path.exists(original_path):
        print(f"File not found: {original_path}")
        return None

    temp_files = []
    attempt = 0
    while attempt < max_retries:
        try:
            silent_video = remove_audio(original_path)
            temp_files.append(silent_video)

            short_filename = f"{uuid.uuid4().hex[:6]}.mp4"
            anonymized_path = os.path.join(os.path.dirname(silent_video), short_filename)
            os.rename(silent_video, anonymized_path)
            temp_files[-1] = anonymized_path  # Track renamed file

            file_size = os.path.getsize(anonymized_path) / (1024 * 1024)
            if file_size > 50:
                print(f"File '{anonymized_path}' is too large ({file_size:.2f}MB). Consider resizing.")
                return None

            file = genai.upload_file(anonymized_path)
            print(f"Uploaded file: {file.uri} (Short name: {short_filename}, ~{file_size:.2f}MB)")
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

    print(f"Final Upload Failure: {original_path}")
    return None


def wait_for_files_active(files):
    """Waits until each uploaded file becomes ACTIVE before using it in a request."""
    for file in files:
        if file is None:
            raise Exception("Some files were not uploaded successfully.")

        while file.state.name == "PROCESSING":
            time.sleep(10)
            file = genai.get_file(file.name)

        if file.state.name != "ACTIVE":
            raise Exception(f"File '{file.name}' failed to process (State: {file.state.name})")


def call_gemini_with_limit(parts):
    """Enforces API rate limit, then calls Gemini with the given prompt parts."""
    global request_times

    current_time = time.time()
    while request_times and request_times[0] < current_time - REQUEST_WINDOW:
        request_times.popleft()

    if len(request_times) >= API_RATE_LIMIT:
        time_to_wait = REQUEST_WINDOW - (current_time - request_times[0])
        print(f"Rate limit reached! Waiting {round(time_to_wait, 2)}s before next request...")
        time.sleep(time_to_wait)

    request_times.append(time.time())
    model = genai.GenerativeModel('models/gemini-2.0-flash')
    return model.generate_content(parts)


def analyze_video_order(video_id, clips):
    """
    Uploads the shuffled clips for a single video,
    waits for them to become ACTIVE, then sends them
    to Gemini with the correct prompt (labels or labels + descriptions).
    Returns the raw text response from Gemini.

    Half the clips are provided as text-only, the rest as video.
    """
    try:
        uploaded_files = {label: upload_to_gemini(path) for label, path in clips.items()}
        wait_for_files_active(uploaded_files.values())

        if USE_DESCRIPTIONS:
            prompt = (
                f"A video has been split into {len(clips)} clips, shuffled randomly. Some clips are provided for some while text description is provided for others.\n"
                "Your task is to analyze each clip deeply to reorder them into the correct temporal sequence. Focus on:\n"
                "1. **Visual content**: Examine the actions, transitions, scene details, and context within each clip.\n"
                "2. **Temporal logic**: Identify the logical progression of events based on what happens before or after.\n"
                "3. **Labels**: Leverage the labels annotation to infer their proper chronological sequence.\n\n"
                "Provide the reordered sequence strictly within order tags and the reason why in this format: "
                "'<order>Label X, Label Y, Label Z, ..., Explanation for order</order>'."
            )
        else:
            prompt = (
                f"A video has been split into {len(clips)} clips, shuffled randomly.\n"
                "Your task is to analyze each clip deeply to reorder them into the correct temporal sequence. Focus on:\n"
                "1. *Visual content*: Examine the actions, transitions, scene details, and context within each clip.\n"
                "2. *Temporal logic*: Identify the logical progression of events based on what happens before or after.\n\n"
                "Provide the reordered sequence strictly within order tags and the reason why in this format: "
                "'<order>Label X, Label Y, Label Z, ..., Explanation for order</order>'."
            )

        parts = [prompt]

        num_clips = len(uploaded_files)
        num_text_only = num_clips // 2
        text_only_keys = random.sample(list(uploaded_files.keys()), num_text_only)

        for label, file_obj in uploaded_files.items():
            if label in text_only_keys:
                parts.append(f"Clip (Text): {label}")
            else:
                base_label = label.split(" - ")[0]
                parts.append(f"Clip (Video): {base_label}")
                parts.append(file_obj)

        response = call_gemini_with_limit(parts)
        if response and response.text:
            return response.text
        else:
            return "ERROR: Empty response"

    except Exception as e:
        return f"ERROR: {str(e)}"


def process_and_save(results, true_orders, output_csv):
    """Extract predictions from results and save to CSV."""
    header = ["Video ID", "Predicted Order", "True Order", "Accuracy (%)", "Full Response"]
    rows = []
    for video_id, predicted_order in results.items():
        order_text = extract_order_tags(predicted_order)
        predicted_labels = re.findall(r"Label ([A-Z]+)", order_text) if order_text else []
        true_order_labels = list(true_orders[video_id].keys())
        accuracy = compute_accuracy(predicted_labels, true_order_labels)
        rows.append([video_id, predicted_labels, true_order_labels, accuracy, predicted_order])
    save_results_csv(rows, output_csv, header)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Gemini 2.0 Flash video reordering inference")
    parser.add_argument("--start", type=int, default=800, help="Start index for dataset slice")
    parser.add_argument("--end", type=int, default=1000, help="End index for dataset slice")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="GeminiPaper5.csv", help="Output CSV path")
    parser.add_argument("--metadata", type=str, default="segment_metadata.json", help="Path to segment metadata JSON")
    parser.add_argument("--api-key", type=str, default=None, help="Gemini API key (defaults to GEMINI_API_KEY env var)")
    args = parser.parse_args()

    random.seed(args.seed)

    api_key = args.api_key or os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)

    video_data, true_orders = load_segment_data(
        json_path=args.metadata, start_index=args.start, end_index=args.end,
        labels=LABELS, use_descriptions=USE_DESCRIPTIONS
    )
    results = {}

    for video_id, clips in video_data.items():
        print(f"--- Analyzing video_id: {video_id} ---")
        inference_result = analyze_video_order(video_id, clips)
        results[video_id] = inference_result
        print(f"Result for {video_id}: {inference_result}\n")

    process_and_save(results, true_orders, args.output)
    print(f"Processed {len(results)} videos. Results saved in '{args.output}'.")


if __name__ == "__main__":
    main()
