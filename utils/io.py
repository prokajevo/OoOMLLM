import os
import csv
import uuid
import subprocess


def remove_audio(input_video):
    """Remove audio track from a video using FFmpeg.

    Args:
        input_video: Path to the input video file.

    Returns:
        Path to the new silent video file.

    Raises:
        FileNotFoundError: If the input video does not exist.
        RuntimeError: If FFmpeg fails to process the video.
    """
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Video file not found: {input_video}")

    anonymized_filename = f"{uuid.uuid4().hex}.mp4"
    output_video = os.path.join(os.path.dirname(input_video), anonymized_filename)

    result = subprocess.run(
        ["ffmpeg", "-i", input_video, "-c:v", "copy", "-an", output_video, "-y"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed to process '{input_video}' (exit code {result.returncode}): "
            f"{result.stderr.decode(errors='replace').strip()}"
        )

    return output_video


def save_results_csv(rows, output_csv, header):
    """
    Append result rows to a CSV file, writing the header only if the file is new.

    Args:
        rows: List of lists, each representing a CSV row.
        output_csv: Path to the output CSV file.
        header: List of column name strings.
    """
    file_exists = os.path.exists(output_csv)

    with open(output_csv, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        for row in rows:
            writer.writerow(row)
