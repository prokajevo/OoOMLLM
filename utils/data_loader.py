import os
import json
import random


def load_metadata(json_path):
    """Load raw metadata from JSON file."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Segment data file not found: {json_path}")
    with open(json_path, "r") as f:
        return json.load(f)


def load_segment_data(json_path="segment_metadata.json", start_index=0, end_index=None,
                      labels=None, use_descriptions=False, filter_ids=None):
    """
    Loads video segment data from a JSON file and returns shuffled clips
    and ground-truth orderings.

    Args:
        json_path: Path to segment_metadata.json.
        start_index: Start index for slicing the dataset.
        end_index: End index for slicing the dataset (None = all).
        labels: List of label strings to assign to segments (e.g. ["BULL", "SPADE", ...]).
        use_descriptions: If True, labels include segment description text.
        filter_ids: Optional set/list of video IDs to include.

    Returns:
        video_clips: dict mapping video_id -> {label_text: file_path} (shuffled).
        true_orders: dict mapping video_id -> {base_label: label_text} (chronological).
            When use_descriptions=False and labels map directly to paths (gemini.py style),
            true_orders maps {base_label: file_path} instead.
    """
    if labels is None:
        raise ValueError("labels must be provided")

    segment_data = load_metadata(json_path)

    if end_index is None:
        end_index = len(segment_data)
    segment_data = segment_data[start_index:end_index]

    if filter_ids:
        filter_set = set(filter_ids)
        segment_data = [v for v in segment_data if v["video_id"] in filter_set]

    video_clips = {}
    true_orders = {}

    for video in segment_data:
        video_id = video["video_id"]
        segments = sorted(video["segments"], key=lambda x: x["part"])

        if use_descriptions:
            # Build label_text with descriptions (gemini2, internVL, LlavaOnevision style)
            to_map = {}
            for i, seg in enumerate(segments):
                base_label = labels[i]
                label_text = f"{base_label} - '{seg['label']}'"
                to_map[base_label] = label_text

            list_of_pairs = []
            for i, seg in enumerate(segments):
                base_label = labels[i]
                label_text = to_map[base_label]
                file_path = seg["output_path"]
                list_of_pairs.append((label_text, file_path))

            random.shuffle(list_of_pairs)
            video_clips[video_id] = {lt: fp for lt, fp in list_of_pairs}
            true_orders[video_id] = to_map
        else:
            # Simple label -> path mapping (gemini.py style)
            true_orders[video_id] = {
                labels[i]: seg["output_path"] for i, seg in enumerate(segments)
            }
            shuffled_items = list(true_orders[video_id].items())
            random.shuffle(shuffled_items)
            video_clips[video_id] = dict(shuffled_items)

        print(f"[load_segment_data] Video {video_id}: True Orders -> {true_orders[video_id]}")
        print(f"[load_segment_data] Video {video_id}: Shuffled Clips -> {video_clips[video_id]}")

    return video_clips, true_orders
