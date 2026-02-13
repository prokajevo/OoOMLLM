import re


def extract_order_tags(response):
    """
    Extract the content within <order>...</order> tags from model response.
    Tries multiple fallback patterns for malformed tags.

    Returns:
        The extracted text, or None if no match found.
    """
    # Primary: well-formed <order>...</order>
    match = re.search(r"<order>(.*?)</order>", response, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: <order>...<order> (missing slash)
    match2 = re.search(r"<order>(.*?)<order>", response, re.DOTALL)
    if match2:
        return match2.group(1).strip()

    # Fallback: </order>...</order> (extra closing tag)
    match3 = re.search(r"</order>(.*?)</order>", response, re.DOTALL)
    if match3:
        return match3.group(1).strip()

    return None


def extract_labels_from_order(order_text, valid_labels):
    """
    Extract label tokens from order text, filtering to only valid labels.

    Args:
        order_text: Raw text from within <order> tags.
        valid_labels: List of valid label strings to match against.

    Returns:
        List of matched labels in order.
    """
    valid_set = set(valid_labels)
    # Remove timestamps like "00:00"
    cleaned = re.sub(r"\d{2}:\d{2}", "", order_text)
    # Split on commas and whitespace
    tokens = [t.strip() for t in re.split(r"[,\s]+", cleaned) if t.strip()]
    return [t for t in tokens if t in valid_set]


def extract_video_numbers(response):
    """
    Extract Video N references from response text.
    Returns list like ["Video 1", "Video 3", "Video 2"].
    """
    matches = re.findall(r'\b(?:Video|video|VIDEO)\s?(\d+)\b', response, re.IGNORECASE)
    return [f"Video {num}" for num in dict.fromkeys(matches)]


def strip_numbers(label):
    """
    Remove a leading number and an optional hyphen (with surrounding spaces)
    from the label. For example, "3 - 'take out the goods'" becomes "'take out the goods'".
    """
    return re.sub(r'^\s*\d+\s*-\s*', '', label)


def compute_accuracy(predicted_labels, true_labels):
    """
    Compute positional accuracy between predicted and true label sequences.

    Args:
        predicted_labels: List of predicted labels in order.
        true_labels: List of true labels in order.

    Returns:
        Accuracy as a float percentage (0-100), rounded to 2 decimals.
    """
    if not true_labels:
        return 0.0
    correct = sum(1 for p, t in zip(predicted_labels, true_labels) if p == t)
    return round((correct / len(true_labels)) * 100, 2)
