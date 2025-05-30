import argparse
import json
import os

from tqdm import tqdm

from models import OpenAIMultimodalModel, GeminiMultimodalModel, QwenMultimodalModel, LlavaMultimodalModel
from models import OpenAIModel
import utils


def main(args):
    llm_type = args.llm_type
    prompt_template_name = args.prompt_template
    path_examples = args.examples
    path_results_dir = args.results_dir
    prefix = args.prefix
    if prefix is None:
        prefix = utils.get_current_time()

    path_frames = os.path.join(os.path.dirname(path_examples), "frames")

    path_output_dir = os.path.join(path_results_dir, prefix)
    utils.mkdir(path_output_dir)

    ###########
    # Get gesture examples
    ###########

    examples = utils.read_json(path_examples)
    print(f"# Examples: {len(examples)}")

    ###########
    # Get a model
    ###########

    prompt_template = utils.read_prompt_template(prompt_template_name)

    if llm_type == "openai":
        model = OpenAIMultimodalModel(model_name="gpt-4o")
    elif llm_type == "gemini":
        model = GeminiMultimodalModel(model_name="gemini-1.5-pro")
    elif llm_type == "qwen":
        model = QwenMultimodalModel(model_name="Qwen/Qwen2.5-VL-7B-Instruct")
    elif llm_type == "llava":
        model = LlavaMultimodalModel(model_name="llava-hf/LLaVA-NeXT-Video-7B-hf")
    else:
        raise Exception(f"Invalid llm_type: {llm_type}")
    model.llm_type = llm_type

    ###########
    # Generate gesture descriptions
    ###########

    generate(
        prompt_template=prompt_template,
        model=model,
        examples=examples,
        path_frames=path_frames,
        path_output_file=os.path.join(path_output_dir, "results.jsonl"),
    )

    print("Done.")


def generate(prompt_template, model, examples, path_frames, path_output_file):

    # # XXX
    # translator = OpenAIModel(model_name="gpt-4o-mini")
    # translation_prompt_template = utils.read_prompt_template("translation_utterances")

    with open(path_output_file, "w") as f:

        for example in tqdm(examples):

            # Get input elements
            utterances = example["corresponding_utterances"]
            video_filename = example["corresponding_video"]
            # physical_perspective_description = example["gesture"]["physical_perspective_description"]
            # gesture_type = example["gesture"]["gesture_type"]

            # Preprocess the input elements
            utterances = preprocess_utterances(utterances=utterances)

            # # XXX
            # trans_prompt = translation_prompt_template.format(input_utterances={"source": utterances.split("\n")})
            # trans_generated_text = translator.generate(prompt=trans_prompt)
            # response = utils.safe_json_loads(generated_text=trans_generated_text, fallback={"target": utterances.split("\n")})
            # utterances = "\n".join(response["target"])

            # Generate a prompt
            prompt = prompt_template.format(utterances=utterances) # default
            # prompt = prompt_template.format(utterances=utterances, physical_perspective_description=physical_perspective_description) # default + physical-perspective description
            # prompt = prompt_template.format(utterances=utterances, gesture_type=gesture_type) # default + gesture type
            # prompt = prompt_template # default without dialogue
            # prompt = prompt_template.format(utterances=utterances) # default without vision

            print("++++++++++++++++++++++")
            print(prompt)
            print("++++++++++++++++++++++")

            generated_text = model.generate(
                prompt=prompt,
                path_frames=path_frames,
                base_filename=video_filename.replace(".mp4", "")
            )

            print("----------------------")
            print(generated_text)
            print("----------------------")

            # Transform to the output format
            result = {
                "example_key": example["example_key"],
                "gesture_type": example["gesture"]["gesture_type"],
                "gold": example["gesture"]["description"],
                "pred": generated_text
            }

            # Save
            json_str = json.dumps(result, ensure_ascii=False)
            f.write(json_str + "\n")


def preprocess_utterances(utterances):
    utterances = [f"{u['speaker']}: {u['utterance']}" for u in utterances]
    utterances = "\n".join(utterances)
    return utterances


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_type", type=str, required=True)
    parser.add_argument("--prompt_template", type=str, default="generation")
    parser.add_argument("--examples", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)
    args = parser.parse_args()
    main(args)
