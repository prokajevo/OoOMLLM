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
import shutil
import base64
import mimetypes
import random
import google.generativeai as genai
import re
import time
import subprocess


# Configure API key (replace with your actual API key)

genai.configure(api_key="") 

import pandas as pd

# Load the CSV file
csv_file_path = "/share/users/student/w/wokajevo/TodOEmpty_Prediction_Videos.csv"
df = pd.read_csv(csv_file_path)

# Extract the list of specific video IDs to process
video_ids_subset = df["Video ID"].tolist()

# Display the first few extracted video IDs
video_ids_subset


# Set random seed for reproducibility
SEED = 46
random.seed(SEED)

# Output file
output_csv = "Gemini1.5FlashVideo4.csv"

LABELS = ["BULL", "SPADE", "HEART", "DIAMOND", "CLUB", "STAR", "MOON", "SUN"]


def load_segment_data(json_path="segment_metadata.json", start_index=0, end_index=None, filter_ids=None):
    """Loads a specified range of video segment data from a JSON file, filtering by a set of video IDs if provided."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Segment data file not found: {json_path}")
    
    with open(json_path, "r") as f:
        segment_data = json.load(f)
    
    # Apply index-based filtering first
    if end_index is None:
        end_index = len(segment_data)
    
    segment_data = segment_data[start_index:end_index]  

    # Apply filtering based on specified video IDs
    if filter_ids:
        segment_data = [video for video in segment_data if video["video_id"] in filter_ids]

    video_clips = {}
    true_orders = {}
    
    for video in segment_data:
        video_id = video["video_id"]
        segments = sorted(video["segments"], key=lambda x: x["part"])  # True order
        true_orders[video_id] = {LABELS[i]: seg["output_path"] for i, seg in enumerate(segments)}
        
        # Keep label-to-path mapping intact but shuffle order for model input
        shuffled_items = list(true_orders[video_id].items())
        random.shuffle(shuffled_items)
        video_clips[video_id] = dict(shuffled_items)
        
        print(f"Video ID: {video_id}")
        print(f"True Order: {true_orders[video_id]}")
        print(f"Shuffled Order for Model: {video_clips[video_id]}")
    
    return video_clips, true_orders



import time

def upload_to_gemini(original_path, mime_type=None, max_retries=5):
    """Uploads a video to Gemini and retries on failure."""
    attempt = 0
    while attempt < max_retries:
        try:
            silent_video = remove_audio(original_path)
            file = genai.upload_file(silent_video, mime_type=mime_type)
            print(f"Uploaded file: {file.uri}")
            return file
        except Exception as e:
            print(f"Upload failed (Attempt {attempt+1}/{max_retries}): {str(e)}")
            time.sleep(2 ** attempt)  # Exponential backoff
            attempt += 1

    raise Exception(f"Failed to upload file after {max_retries} attempts")


def remove_audio(input_video):
    """Removes audio from the video using FFmpeg."""
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Video file not found: {input_video}")

    anonymized_filename = f"{uuid.uuid4().hex}.mp4"
    output_video = os.path.join(os.path.dirname(input_video), anonymized_filename)

    command = ["ffmpeg", "-i", input_video, "-c:v", "copy", "-an", output_video, "-y"]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return output_video



def wait_for_files_active(files):
    """Waits for uploaded files to become ACTIVE before using them."""
    print("Waiting for file processing...")
    for name in (file.name for file in files):
        file = genai.get_file(name)
        while file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(10)  # Poll every 10 seconds
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


def analyze_video_order(video_id, clips):
    """
    Uploads video clips (with randomized names), waits for processing, and sends them to Gemini.

    Args:
        video_id (str): Unique identifier for the video.
        clips (dict): Dictionary mapping clip labels to file paths.

    Returns:
        str: The reordered sequence returned by Gemini.
    """

    #model = genai.GenerativeModel('models/gemini-1.5-flash')

    try:
        # ✅ No need to shuffle again; clips are already shuffled in `load_segment_data`
        uploaded_files = [upload_to_gemini(path) for _, path in clips.items()]
        wait_for_files_active(uploaded_files)

        # Construct a structured prompt
              
        prompt = (
            f"A video has been split into {len(clips)} clips and shuffled randomly.\n"
            "Your task is to analyze each clip deeply and reorder them into the correct temporal sequence.\n"
            "Focus on:\n"
            "1. *Visual content*: Examine the actions, transitions, scene details, and context within each clip.\n"
            "2. *Temporal logic*: Identify the logical progression of events based on what happens before or after.\n\n"
            "Provide the reordered sequence strictly within order tags in this format: '<order>Video X , Video Y, ...</order>'."
        )

        """
        prompt = (
            f"The video has been split into {len(clips)} clips in random order. "
            "Reconstruct the original timeline by analyzing and connecting the clips. "
            "Think through this step-by-step:\n\n"
            
            "1. **Individual Clip Analysis**\n"
            "   - Describe key elements in each clip: actions (e.g., 'person pouring coffee'), scene details (e.g., 'intact glass on table'), "
            "transitions (e.g., 'sunset to night'), and contextual clues (e.g., 'half-filled notebook')\n"
            "   - Note temporal markers: time of day, progress bars, character positions, object states\n\n"
            
            "2. **Comparative Analysis**\n"
            "   - Identify relationships between clips using:\n"
            "     a) Action continuity (e.g., unwrapped gift → opened gift)\n"
            "     b) Scene evolution (e.g., clean kitchen → dirty dishes)\n"
            "     c) Logical cause/effect (intact window → broken window)\n"
            "     d) Narrative flow (establishing shot → close-up)\n\n"
            
            "3. **Timeline Construction**\n"
            "   - Create a progression chain using phrases like:\n"
            "     'Clip X must come before Clip Y because...'\n"
            "     'Clip Z shows a consequence that requires...'\n"
            "   - Resolve ambiguities by prioritizing strong causal links over weak associations\n\n"
            
            "4. **Final Verification**\n"
            "   - Walk through your proposed sequence to check for:\n"
            "     - Continuous spatial logic\n"
            "     - Consistent object states\n"
            "     - Natural action progression\n\n"
            
            "Present your answer as:\n"
            "REASONING: [Detailed analysis using steps 1-4 above]\n"
            "Provide the reordered sequence strictly within order tags in this format: '<order>Video X , Video Y, ...</order>'"
        )""" 

        # Prepare request parts
        parts = [prompt]

        # Append file references to the prompt
        for label, file in zip(clips.keys(), uploaded_files):
            parts.append(f"Clip label: {label}")
            parts.append(file)  # Append only the file object, no metadata

        #print(parts)

        # Generate response
        response = call_gemini_with_limit(parts)
        
        if response and response.text:
            print(f"✅ Model Response for {video_id}: {response.text}")
            return response.text
        else:
            print(f"⚠️ Empty response received for {video_id}.")
            return "ERROR: Empty response"

    except Exception as e:
        print(f"❌ Error analyzing video {video_id}: {str(e)}")
        return f"ERROR: {str(e)}"


from collections import deque

API_RATE_LIMIT = 10  # 10 requests per minute
REQUEST_WINDOW = 60  # 60 seconds
request_times = deque()  # Stores timestamps of past requests

def call_gemini_with_limit(parts):
    """Dynamically ensures we do not exceed 10 requests per minute."""
    global request_times

    # Remove timestamps older than 60 seconds
    current_time = time.time()
    while request_times and request_times[0] < current_time - REQUEST_WINDOW:
        request_times.popleft()

    # If we're at the limit, wait
    if len(request_times) >= API_RATE_LIMIT:
        time_to_wait = REQUEST_WINDOW - (current_time - request_times[0])
        print(f"Rate limit reached! Waiting {round(time_to_wait, 2)}s before next request...")
        time.sleep(time_to_wait)

    # Make the request
    request_times.append(time.time())
    model = genai.GenerativeModel('models/gemini-1.5-flash')
    return model.generate_content(parts)

import re

def clean_predicted_order(predicted_order):
    """Extracts and cleans predicted order, removing timestamps and unwanted text."""
    match = re.search(r"<order>(.*?)</order>", predicted_order, re.IGNORECASE | re.DOTALL)
    if match:
        extracted_text = match.group(1).strip()

        # Remove timestamps like "00:00" using regex
        cleaned_order = re.sub(r"00:\d{2}", "", extracted_text)

        # Split into list and clean spaces
        predicted_labels = [p.strip() for p in re.split(r"[,\s]+", cleaned_order) if p.strip() in LABELS]

        return predicted_labels
    return []

def compute_accuracy(predicted_order, true_order):
    """Computes accuracy by comparing the cleaned predicted order against the true order."""
    predicted_order_clean = clean_predicted_order(predicted_order)
    
    print(f"🔍 Processed Predicted Order: {predicted_order_clean}")
    print(f"✅ True Order: {true_order}")

    correct = sum(1 for pred, true in zip(predicted_order_clean, true_order) if pred == true)
    return round((correct / len(true_order)) * 100, 2) if true_order else 0.0, predicted_order_clean


X_SAVE_INTERVAL = 10  # Save every 5 processed videos (adjust as needed)

def save_results_incrementally(results, true_orders, output_csv):
    """Appends results incrementally to the CSV file instead of saving everything at the end."""
    file_exists = os.path.exists(output_csv)
    
    with open(output_csv, mode='a', newline='') as file:  # 'a' mode for appending
        writer = csv.writer(file)
        
        # Write header only if file is new
        if not file_exists:
            writer.writerow(["Video ID", "Predicted Order", "True Order", "Accuracy (%)", "Inference Return"])
        
        for video_id, predicted_order in results.items():
            true_order_labels = list(true_orders[video_id].keys())
            accuracy, predicted_order_clean = compute_accuracy(predicted_order, true_order_labels)
            print(f"Writing to CSV: {video_id}, {predicted_order}, {true_order_labels}, {accuracy}")
            writer.writerow([video_id, predicted_order_clean, ", ".join(true_order_labels), accuracy, predicted_order])

def main():
    """Main execution loop with periodic CSV saving."""
    try:
        start_index = 3000  # Set desired range start
        end_index = 3580   # Set desired range end (None for all data)

        video_data, true_orders = load_segment_data(start_index=start_index, end_index=end_index)
        results = {}
        
        for i, (video_id, clips) in enumerate(video_data.items(), 1):
            result = analyze_video_order(video_id, clips)
            print(f"Model Output for {video_id}: {result}")
            results[video_id] = result

            # Save results every X iterations
            if i % X_SAVE_INTERVAL == 0:
                print(f"💾 Saving results at iteration {i}")
                save_results_incrementally(results, true_orders, output_csv)
                results.clear()  # Clear stored results after saving to avoid duplicate writes

        # Final save for any remaining results
        if results:
            print("💾 Final saving of remaining results.")
            save_results_incrementally(results, true_orders, output_csv)

    except Exception as e:
        print(f"❌ Error: {str(e)}")

# Run main function
if __name__ == "__main__":
    main()