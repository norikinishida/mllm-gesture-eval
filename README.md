# mllm-gesture-eval

This repository contains the code and gesture type annotations (coming soon) used in our paper:
**"Do Multimodal Large Language Models Truly See What We Point At? Investigating Indexical, Iconic, and Symbolic Gesture Comprehension"** (ACL 2025, *to appear*)

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

## API Key Configuration

Some scripts require access to external APIs (e.g., OpenAI, Gemini).  
Before running them, please create a file named `keys.json` in the `codes/` directory, and include your API keys in the following format:

```json
{
  "openai": "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
  "gemini": "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
}
```

Each script will automatically load the required key based on the --llm_type argument (e.g., openai, gemini, qwen).

🔐 Note: Do not share your `keys.json` publicly. It is listed in `.gitignore` by default.

## Usage

1. **Prepare the dataset** (run from inside the `codes/dataset-preparation/` directory):

   ```bash
   cd codes/dataset-preparation
   ./run.sh # COMING SOON
   ```

2. **Generate gesture descriptions** (run from inside the `codes/` directory):

   ```bash
   cd codes
   python step1_generate.py --llm_type ${LLM_TYPE} --examples ${EXAMPLES} --results_dir ${RESULTS_DIR} --prefix ${MY_PREFIX}
   ```

   The script requires you to specify several environment variables:

   * `EXAMPLES`: Path to the input preprocessed dataset (JSON file)
   * `RESULTS_DIR`: Directory where results will be stored
   * `LLM_TYPE`: Backend MLLM to use (e.g., `openai`, `gemini`, `qwen`, `llava`)
   * `MY_PREFIX`: Identifier for the experiment version

   Example:

   ```bash
   EXAMPLES=<path to this repository>/data/mscc/v1/examples.json
   RESULTS_DIR=<path to this repository>/results
   LLM_TYPE=openai
   MY_PREFIX=hoge
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
    booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL)},
    year={2025}
}
```

## Contact

For any questions or issues, please contact:
**Noriki Nishida** – nishidanoriki\[at]riken.jp
