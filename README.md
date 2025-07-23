# mllm-gesture-eval

This repository contains the code and gesture type annotations (coming soon) used in our paper:
**"Do Multimodal Large Language Models Truly See What We Point At? Investigating Indexical, Iconic, and Symbolic Gesture Comprehension"** (Nishida et al., ACL 2025).

The repository provides:

* Preprocessing scripts for the dataset used in our experiments
* Step-by-step code and prompt templates for running gesture explanation tasks with multimodal large language models (MLLMs), including:
  - [GPT-4o](https://platform.openai.com/docs/models/gpt-4o) (OpenAI)
  - [Gemini 1.5 Pro](https://deepmind.google/models/gemini/pro) (Google DeepMind)
  - [Qwen2.5-VL-7B-Instruct](https://github.com/QwenLM/Qwen2.5-VL) (Alibaba)
  - [LLaVA-NeXT-Video](https://github.com/LLaVA-VL/LLaVA-NeXT) (LLaVA Project)
* Evaluation scripts (llm-as-a-judge)

## Repository Structure

```
.
├── README.md                     # This file
├── codes/                        # All scripts and templates for processing and evaluation
│   ├── dataset-preparation/     # Preprocessing scripts for Miraikan SC Corpus
│   ├── step1_generate.py        # Run LLMs to generate gesture descriptions
│   ├── step2_evaluate_by_llm.py # Use LLMs to evaluate generated descriptions
│   ├── step3_calc_evaluation_scores.py # Aggregate and score the evaluations
│   ├── utils.py                 # Utility functions
│   ├── prompt_templates/        # Prompt templates
│   └── run.sh                   # Example script for running the full pipeline
├── data/                         # Directory for input data (COMING SOON)
├── results/                      # Directory for generated descriptions and evaluation results
```

## Requirements

We recommend using the latest version of Python (>=3.10) and installing dependencies via:

```bash
pip install -r requirements.txt
```

## Dataset Format

The dataset used in our experiments is formatted as a JSON list, where each entry corresponds to a gesture instance. Each entry contains:
- the gesture annotations,
- the utterances overlapping with the gesture time span, and
- the video clip overlapping with the gesture time span.

Here is an example entry:
```json
{
  "example_key": "data04_264.6_266.935",
  "corresponding_utterances": [
    {
      "id": "a7418",
      "time_span": [264.812, 266.511],
      "speaker": "scA",
      "utterance": "こんなでかい天文台,知ってます?"
    },
    ...
  ],
  "gesture": {
    "id": "a8042",
    "time_span": [264.6, 266.935],
    "gesturer": "scA",
    "position": "hand",
    "perspective": "intentional",
    "description": "客に対し，模型への注意を向けさせ，「天文台」が指示する対象を明示する",
    "gesture_type": "Indexical"
  }
}
```

### Field Descriptions:

- `example_key` (str): Unique identifier for the gesture instance
- `corresponding_utterances` (list[dict]): List of utterance objects, each containing:
  - `id` (str): Unique ID for the utterance (e.g., ELAN annotation ID)
  - `time_span` (tuple[float]): Start and end time in seconds
  - `speaker` (str): Speaker label (e.g., `v01`, `scA`)
  - `utterance` (str): Transcribed text of the utterance
- `gesture` (dict): Object containing gesture annotations:
  - `id` (str): Unique ID for the gesture annotation (e.g., ELAN annotation ID)
  - `time_span` (tuple[float]): Start and end time of the gesture in seconds
  - `gesturer` (str): Identifier of the person performing the gesture (e.g., `scA`)
  - `position` (str): Body part used in the gesture (e.g., `hand`)
  - `perspective` (str): Indicates the perspective used to write the `description`: (e.g., `intentional`)
  - `description` (str): Gesture description (i.e., *relevant annotation*)
  - `gesture_type` (str): Gesture types (e.g., `Indexical`, `Iconic`, `Symbolic`)

### Frame Image Layout:

Each gesture instance is also associated with a directory of extracted frame images.  
The expected directory structure is:


```
<dataset_dir>/
├── dataset.json
└── frames/
    └── <example_key>/
        ├── <example_key>.frame_000.jpg
        ├── <example_key>.frame_001.jpg
        ├── <example_key>.frame_002.jpg
        └── ...
```

For example, if `example_key = "data05_449.382_453.557"`, the expected frame image files are located at:

```
<dataset_dir>/
├── dataset.json
└── frames/
    └── data05_449.382_453.557/
        ├── data05_449.382_453.557.frame_000.jpg
        ├── data05_449.382_453.557.frame_001.jpg
        ├── data05_449.382_453.557.frame_002.jpg
        └── ...
```

These frames are used as visual input to multimodal LLMs.

### ⚠️ Dataset Availability:

Miraikan SC Corpus, the dataset used in our paper, is currently undergoing ethical review and preparation for public release.  
We plan to make the dataset available upon completion of this process.

In the meantime, you may prepare your own dataset in the same format and run the full pipeline as described.  
To use your own dataset, format the JSON and frame directory as described above and place them under:

```
data/your_dataset_name/
├── dataset.json
└── frames/
```

You can then follow the steps in the [Usage](#usage) section (from Step 2) to run the full evaluation pipeline.

## Usage

1. **Prepare the dataset** (run from inside the `codes/dataset-preparation/` directory):

   ```bash
   cd codes/dataset-preparation
   ./run.sh # COMING SOON
   ```

2. **Generate gesture descriptions** (run from inside the `codes/` directory):

   ```bash
   cd codes
   python step1_generate.py --llm_type ${LLM_TYPE} --dataset ${DATASET} --results_dir ${RESULTS_DIR} --prefix ${MY_PREFIX}
   ```

   The script requires you to specify several environment variables:

   * `LLM_TYPE`: Backend MLLM to use (e.g., `openai`, `gemini`, `qwen`, `llava`)
   * `DATASET`: Path to the input preprocessed dataset (JSON file)
   * `RESULTS_DIR`: Directory where results will be stored
   * `MY_PREFIX`: Identifier for the experiment version

   Example:

   ```bash
   LLM_TYPE=openai
   DATASET=<path to this repository>/data/mscc/v1/dataset.json
   RESULTS_DIR=<path to this repository>/results
   MY_PREFIX=example
   ```

3. **Evaluate the explanations using LLMs** (also from inside the `codes/` directory):

   ```bash
   python step2_evaluate_by_llm.py --input_file ${RESULTS_DIR}/${MY_PREFIX}/results.jsonl --output_file ${RESULTS_DIR}/${MY_PREFIX}/evaluation_by_llm.jsonl
   ```

4. **Calculate evaluation scores** (also from inside the `codes/` directory):

   ```bash
   python step3_calc_evaluation_scores.py --input ${RESULTS_DIR}/${MY_PREFIX}/evaluation_by_llm.jsonl
   ```

Or, run the whole pipeline (from inside `codes/`):

```bash
cd codes
./run.sh
```

## License

This project is licensed under the MIT License.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{nishida2025domultimodal,
    title={Do Multimodal Large Language Models Truly See What We Point At? Investigating Indexical, Iconic, and Symbolic Gesture Comprehension},
    author={Nishida, Noriki and,
            Inoue, Koji and
            Nakayama, Hideki and
            Bono, Mayumi and
            Takanashi, Katsuya},
    booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025)},
    year={2025}
}
```

## Contact

For any questions or issues, please contact:
**Noriki Nishida** – noriki.nishida\[at]riken.jp
