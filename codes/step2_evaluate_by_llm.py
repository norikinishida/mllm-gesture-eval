import argparse
import json

from models import OpenAIModel
from tqdm import tqdm

import utils


def main(args):
    prompt_template_name = args.prompt_template
    path_input_file = args.input_file
    path_output_file = args.output_file

    dataset = []
    with open(path_input_file) as f:
        for line in f.readlines():
            example = json.loads(line.strip())
            dataset.append(example)
    print(f"# Examples: {len(dataset)}")

    prompt_template = utils.read_prompt_template(prompt_template_name)

    model = OpenAIModel(model_name="gpt-4o-mini")

    evaluate(
        prompt_template=prompt_template,
        model=model,
        dataset=dataset,
        path_output_file=path_output_file,
    )

    print("Done.")


def evaluate(prompt_template, model, dataset, path_output_file):

    # # XXX
    # translator = OpenAIModel(model_name="gpt-4o-mini")
    # translation_prompt_template = utils.read_prompt_template("translation_reference")

    with open(path_output_file, "w") as f:

        for example in tqdm(dataset):

            # Get input elements
            gesture_type = example["gesture_type"]
            gold = example["gold"]
            pred = example["pred"]

            # # XXX
            # trans_prompt = translation_prompt_template.format(input_reference={"source": gold})
            # trans_generated_text = translator.generate(prompt=trans_prompt)
            # response = utils.safe_json_loads(generated_text=trans_generated_text, fallback={"target": gold})
            # gold = response["target"]

            # Generate a prompt
            prompt = prompt_template.format(gesture_type=gesture_type, gold=gold, pred=pred)

            print("++++++++++++++++++++++")
            print(prompt)
            print("++++++++++++++++++++++")

            # Generate an evaluation result
            generated_text = model.generate(prompt=prompt)

            print("----------------------")
            print(generated_text)
            print("----------------------")

            # Transform to the output format
            result = example | {"evaluation": generated_text}

            # Save
            json_str = json.dumps(result, ensure_ascii=False)
            f.write(json_str + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_template", type=str, default="evaluation")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    main(args)

