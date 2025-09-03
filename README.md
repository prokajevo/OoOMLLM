# Out of Order: Evaluating MLLMs on Reordering Shuffled Video Segments

[![Paper Status: Under Review](https://img.shields.io/badge/Paper%20Status-Under%20Review%20(EMNLP%202025)-blue)](https://openreview.net/forum?id=O5qq8NQbL2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official code and resources for the Master's thesis, "Out of Order: Evaluating MLLMs on Reordering Shuffled Video Segments, Temporal Logic, and Multimodal Event Understanding."

This research confronts the "evaluation deficit" in video understanding by introducing a novel video segment reordering task. We present **SPLICE** (Sequential Processing for Learning and Inference in Chronological Events), a human-curated benchmark designed to rigorously test an MLLM's grasp of temporal logic, causality, and multimodal event structure, moving beyond simple classification to probe deep, structural reasoning.

---

## 📜 Abstract

The rapid advancement of Multimodal Large Language Models (MLLMs) has pushed capabilities into the complex domain of video understanding. However, current benchmarks often fail to robustly assess a model's grasp of temporal logic, being susceptible to linguistic shortcuts or focusing on simple classification. This thesis introduces a novel evaluation methodology: a video segment reordering task instantiated through the **SPLICE** benchmark.

SPLICE is a human-curated benchmark derived from 3,381 instructional videos from the COIN dataset, segmented into 11,423 coherent event clips. Our extensive evaluation of leading MLLMs (including the Gemini and Qwen families) on SPLICE reveals a substantial performance gap. The best-performing model achieves a perfect sequence match accuracy of only **51%**, compared to a human baseline of approximately **85%**.

Crucially, results show that while textual annotations significantly improve model performance, they have no effect on human accuracy, indicating a strong reliance on language priors over genuine visual understanding in current MLLMs. This work not only quantifies the limitations of MLLMs but also validates the reordering task as a rigorous diagnostic tool for driving future progress in building more capable AI systems.

## 📊 Key Findings

1.  **Significant Human-Model Performance Gap**: Even state-of-the-art models lag significantly behind the human baseline of 84.9% binary accuracy. The best model, Gemini-2.0-Flash-Exp, scored 51.1%.

2.  **Over-reliance on Text**: Models show a dramatic performance increase when provided with text descriptions ("Video+Text" modality), while human performance remains unaffected. This suggests models use text as a "linguistic shortcut," bypassing deep visual reasoning.

3.  **Performance Degrades with Complexity**: Model accuracy drops sharply as the number of video segments to reorder increases, indicating a struggle with maintaining long-range temporal coherence. Humans exhibit a much more graceful degradation.

    
    *Figure 5.1 from the thesis: Binary accuracy versus the number of clip segments.*

4.  **Weakness in Contextual & Spatial Reasoning**: Models perform relatively well on tasks driven by clear causal/temporal logic ("Make" tasks, ~68%) but fail catastrophically on tasks requiring contextual reasoning ("Change/Replace" tasks, ~32%) due to a strong *Visual Similarity Bias*. Spatial reasoning remains a profound weakness.

### Performance Summary

| Model                  | Vision Only (Binary) | Vision+Text (Binary) | Text Only (Binary) |
| ---------------------- | -------------------- | -------------------- | ------------------ |
| **Human Baseline**     | **0.8486**           | **0.8332**           | -                  |
| Gemini-2.0-Flash-Exp   | 0.5108               | 0.6939               | 0.5271             |
| Gemini-1.5-Flash       | 0.4599               | 0.5936               | 0.4642             |
| Qwen2-VL-72B           | 0.2990               | 0.5708               | 0.5402             |
| InternVL2.5-78B        | 0.2899               | 0.4856               | 0.4768             |
| LLaVA-OneVision-72B    | 0.2260               | 0.4256               | 0.4210             |
| **Random Baseline**    | **0.2114**           | **0.2114**           | -                  |

## 🗂️ The SPLICE Benchmark

The SPLICE benchmark is the core contribution of this work. It was created through a rigorous, multi-stage curation pipeline.

-   **Source**: COIN Dataset (Comprehensive Instructional Video Analysis)
-   **Size**: 3,381 validated videos, 11,423 event clips
-   **Task**: Reorder a set of shuffled, anonymized video clips from a single event into their correct chronological sequence.
-   **Annotation**: Event-based segmentation based on ground-truth timestamps from COIN.
-   **Validation**: Each task was validated by human annotators to ensure solvability and remove ambiguity.

The benchmark is designed to probe five key dimensions of reasoning:
1.  **Temporal Reasoning**: Tracking object states across time.
2.  **Causal Reasoning**: Inferring cause-effect relationships.
3.  **Contextual Reasoning**: Understanding environmental and process dependencies.
4.  **Spatial Reasoning**: Interpreting trajectories and orientations.
5.  **Commonsense Reasoning**: Applying prior world knowledge.

## ⚙️ Repository Structure

```
.
├── gemini.py                   # Inference script for Gemini 1.5 Flash API
├── gemini2.py                  # Inference script for Gemini 2.0 Flash API (experimental)
├── internVL.py                 # Inference script for InternVL 2.5
├── LlavaOnevision.py           # Inference script for LLaVA-OneVision
├── Qwen 2_VL.py                # Inference script for Qwen2-VL
├── segment_metadata.json       # Metadata file with video paths and labels
└── requirements.txt            # Python dependencies
```

## 🚀 Getting Started

### 1. Installation

Clone the repository and install the required dependencies.

```bash
git clone https://github.com/prokajevo/OoOMLLM.git
cd out-of-order-mllms
pip install -r requirements.txt
```

### 2. Dataset

The SPLICE benchmark dataset is central to running these experiments. The `segment_metadata.json` file in this repository contains the paths and ground-truth information.

-   **Sample Data**: A sample of the benchmark can be found here: `[Placeholder for Google Drive link]`
-   **Full Dataset**: The full dataset will be made publicly available upon publication of the research paper. Please stay tuned for updates.

Download the video clips and ensure the paths in `segment_metadata.json` correspond to their location on your local machine.

### 3. API Keys

For experiments involving the Gemini models, you must configure your Google AI API key.

```python
# In gemini.py and gemini2.py
genai.configure(api_key="YOUR_GEMINI_API_KEY")
```

## ▶️ Reproducing Experiments

Each script is designed to run inference for a specific model on a subset of the SPLICE benchmark. The scripts can be configured by changing the `START_INDEX` and `END_INDEX` variables internally.

**Example: Running the LLaVA-OneVision experiment**

1.  Open `LlavaOnevision.py`.
2.  Set the `START_INDEX` and `END_INDEX` to define the slice of the dataset you want to process.
3.  Ensure the model name (`llava-hf/llava-onevision-qwen2-72b-ov-hf`) is correct and that you have sufficient GPU memory.
4.  Run the script from the command line:

```bash
python LlavaOnevision.py
```

The script will:
-   Load the segment data from `segment_metadata.json`.
-   For each video, load the shuffled clips.
-   Preprocess the video frames and construct the prompt.
-   Run inference to get the model's predicted order.
-   Save the results to a CSV file (e.g., `OOVVLABEL72B_7.csv`).

The same workflow applies to `internVL.py` and `Qwen 2_VL.py`. The Gemini scripts (`gemini.py`, `gemini2.py`) will additionally handle file uploads and API rate limiting.


```
## 🙏 Acknowledgments


This repository provides the official code implementation accompanying the research conducted for the Master's thesis of Wilfred Okajevo, “Out of Order: Evaluating MLLMs on Reordering Shuffled Video Segments, Temporal Logic, and Multimodal Event Understanding,” submitted in fulfillment of the requirements for the degree of Master of Science in Cognitive Science at Universität Osnabrück, under the supervision of Dr. Mohamad Ballout and Prof. Dr. Elia Bruni.
