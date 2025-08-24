#!/usr/bin/env python3
#SBATCH --job-name="Gemini"
#SBATCH --time=48:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=GeminiPaper5.txt

import os
import uuid
import json
import csv
import time
import re
import random
import subprocess
from collections import deque

import google.generativeai as genai

# -------------------------------------------------------------------
# ----------------------- CONFIGURATIONS ----------------------------
# -------------------------------------------------------------------

# ✅ (1) Toggle whether to use descriptions with labels
USE_DESCRIPTIONS = True

# ✅ (2) Output CSV file path
output_csv = "GeminiPaper5.csv"

# ✅ (3) API Rate Limit Settings
API_RATE_LIMIT = 10    # 10 requests per minute
REQUEST_WINDOW = 60    # 60 seconds
request_times = deque()

# ✅ (4) Label set for indexing segments
LABELS = ["BULL", "SPADE", "HEART", "DIAMOND", "CLUB", "STAR", "MOON", "SUN"]

# ✅ (5) Gemini API Key (Replace with your own key!)
genai.configure(api_key="") # Add your API key here



# -------------------------------------------------------------------
# ------------------------- CORE FUNCTIONS --------------------------
# -------------------------------------------------------------------

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

        # Build "true order" mapping from just the labels (for final accuracy check)
        # e.g. { "BULL" -> "BULL - 'insert money...'", "SPADE" -> "SPADE - 'press button'", ... }
        to_map = {}
        for i, seg in enumerate(segments):
            base_label = LABELS[i]  # e.g. "BULL"
            if USE_DESCRIPTIONS:
                label_text = f"{base_label} - '{seg['label']}'"
            else:
                label_text = base_label
            
            to_map[base_label] = label_text  # for accuracy checking in order

        # Now build a list of (label_text, file_path) for shuffling
        list_of_pairs = []
        for i, seg in enumerate(segments):
            base_label = LABELS[i]
            # We'll use the already-created text
            label_text = to_map[base_label]     # e.g. "BULL - 'insert money...'"
            file_path = seg["output_path"]      # actual path e.g. "video_segments/oDAY5PMwZgU_part_1.mp4"
            list_of_pairs.append((label_text, file_path))

        # Shuffle the pairs
        random.shuffle(list_of_pairs)

        # Convert to dictionary for model: { "BULL - 'desc'" -> "video_segments/..." }
        shuffled_dict = {label_text: file_path for (label_text, file_path) in list_of_pairs}

        # Save results
        true_orders[video_id] = to_map       # keep correct chronological labels for accuracy
        video_clips[video_id] = shuffled_dict

        # For debugging, see partial results
        print(f"[load_segment_data] Video {video_id}: True Orders -> {true_orders[video_id]}")
        print(f"[load_segment_data] Video {video_id}: Shuffled Clips -> {video_clips[video_id]}")

    return video_clips, true_orders


def remove_audio(input_video):
    """
    Removes audio from the video using FFmpeg 
    to reduce file size and anonymize audio.
    """
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
        print(f"❌ File not found: {original_path}")
        return None

    attempt = 0
    while attempt < max_retries:
        try:
            # Remove audio track to minimize size
            silent_video = remove_audio(original_path)

            # Rename to anonymize
            short_filename = f"{uuid.uuid4().hex[:6]}.mp4"
            anonymized_path = os.path.join(os.path.dirname(silent_video), short_filename)
            os.rename(silent_video, anonymized_path)

            # Optional: Check file size (in MB)
            file_size = os.path.getsize(anonymized_path) / (1024 * 1024)
            if file_size > 50:
                print(f"⚠️ File '{anonymized_path}' is too large ({file_size:.2f}MB). Consider resizing.")
                return None

            # Attempt upload
            file = genai.upload_file(anonymized_path)
            print(f"✅ Uploaded file: {file.uri} (Short name: {short_filename}, ~{file_size:.2f}MB)")
            return file

        except Exception as e:
            print(f"⚠️ Upload failed (Attempt {attempt+1}/{max_retries}): {str(e)}")
            time.sleep(2 ** attempt)  # Exponential backoff
            attempt += 1

    print(f"❌ Final Upload Failure: {original_path}")
    return None


def wait_for_files_active(files):
    """
    Waits until each uploaded file becomes ACTIVE before using it in a request.
    """
    for file in files:
        if file is None:
            # If the file is None, it never uploaded successfully
            raise Exception("❌ Some files were not uploaded successfully.")

        while file.state.name == "PROCESSING":
            time.sleep(10)
            file = genai.get_file(file.name)

        if file.state.name != "ACTIVE":
            raise Exception(f"❌ File '{file.name}' failed to process (State: {file.state.name})")


def call_gemini_with_limit(parts):
    """
    Enforces API rate limit, then calls Gemini with the given prompt parts.
    """
    global request_times

    current_time = time.time()
    # Remove old timestamps
    while request_times and request_times[0] < current_time - REQUEST_WINDOW:
        request_times.popleft()

    # Check rate limit
    if len(request_times) >= API_RATE_LIMIT:
        time_to_wait = REQUEST_WINDOW - (current_time - request_times[0])
        print(f"⏳ Rate limit reached! Waiting {round(time_to_wait, 2)}s before next request...")
        time.sleep(time_to_wait)
    print(parts)

    request_times.append(time.time())
    model = genai.GenerativeModel('models/gemini-2.0-flash')
    return model.generate_content(parts)


def analyze_video_order(video_id, clips):
    """
    Uploads the shuffled clips for a single video,
    waits for them to become ACTIVE, then sends them
    to Gemini with the correct prompt (labels or labels + descriptions).
    Returns the raw text response from Gemini.
    
    Experimental modification:
    If the video has exactly 2 or 3 clips, a random clip is provided as text-only,
    while the remaining clip(s) are attached as video files.
    """
    try:
        # Upload each clip in the shuffled dictionary
        uploaded_files = {label: upload_to_gemini(path) for label, path in clips.items()}

        # Ensure all files are active
        wait_for_files_active(uploaded_files.values())


        # Pick the prompt based on description usage
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

        # Build parts for Gemini

        parts = [prompt]

                # Attach each label + file reference with modality adjustments:
        # Attach each label + file reference with modality adjustments:
        num_clips = len(uploaded_files)
        num_text_only = num_clips // 2  # Calculate how many should be text only
        # Randomly select keys to be text only
        text_only_keys = random.sample(list(uploaded_files.keys()), num_text_only)

        for label, file_obj in uploaded_files.items():
            if label in text_only_keys:
                # Text-only modality: include full label with description even if descriptions are globally disabled
                parts.append(f"Clip (Text): {label}")
            else:
                # Video modality: strip description (keep only the base label before " - ")
                base_label = label.split(" - ")[0]
                parts.append(f"Clip (Video): {base_label}")
                parts.append(file_obj)

        # Call Gemini
        response = call_gemini_with_limit(parts)
        if response and response.text:
            return response.text
        else:
            return "ERROR: Empty response"

    except Exception as e:
        return f"ERROR: {str(e)}"


def compute_accuracy(predicted_order, true_order):
    """
    Extracts the predicted order from Gemini's response and compares 
    it with the keys of `true_order` to compute a simple percentage accuracy.
    """
    # Try to find <order> ... </order> block
    match = re.search(r"<order>(.*?)</order>", predicted_order, re.IGNORECASE | re.DOTALL)
    if match:
        extracted_text = match.group(1).strip()
        # Extract only the label portion before " - "
        predicted_labels = re.findall(r"Label ([A-Z]+)", extracted_text)
    else:
        predicted_labels = []

    # Compare predicted_labels vs. the actual order (keys of true_order)
    correct = sum(1 for pred, true_lbl in zip(predicted_labels, true_order.keys()) if pred == true_lbl)
    accuracy = round((correct / len(true_order)) * 100, 2) if true_order else 0.0

    return accuracy, predicted_labels


def save_results(results, true_orders):
    """
    Saves the inference results into a CSV file. 
    Each row includes:
      - Video ID
      - Predicted (extracted) labels
      - True order (labels)
      - Accuracy
      - Full raw response
    """
    file_exists = os.path.exists(output_csv)

    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header only if file is new
        if not file_exists:
            writer.writerow(["Video ID", "Predicted Order", "True Order", "Accuracy (%)", "Full Response"])

        for video_id, predicted_order in results.items():
            # Keys of true_orders[video_id] represent the correct order in label terms
            accuracy, predicted_labels = compute_accuracy(predicted_order, true_orders[video_id])
            true_order_labels = list(true_orders[video_id].keys())  # e.g., ["BULL", "SPADE", ...]

            writer.writerow([
                video_id,
                predicted_labels,
                true_order_labels,
                accuracy,
                predicted_order
            ])


def main():
    """
    Main entry point. 
    - Prompts user for start/end indices.
    - Loads segment data.
    - Analyzes each video in range.
    - Saves results to CSV.
    """
    # Let user specify portion of dataset
    start_index = 800
    end_index = 1000

    video_data, true_orders = load_segment_data(start_index=start_index, end_index=end_index)
    results = {}

    # Process each video
    for video_id, clips in video_data.items():
        print(f"--- Analyzing video_id: {video_id} ---")
        inference_result = analyze_video_order(video_id, clips)
        results[video_id] = inference_result
        print(f"Result for {video_id}: {inference_result}\n")

    # Save final results
    save_results(results, true_orders)
    print(f"✅ Processed {len(results)} videos. Results saved in '{output_csv}'.")


if __name__ == "__main__":
    main()