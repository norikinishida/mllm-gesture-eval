import datetime
import io
import json
import os


def get_current_time():
    return datetime.datetime.now().strftime("%b%d_%H-%M-%S")


def read_json(path, encoding=None):
    if encoding is None:
        with open(path) as f:
            dct = json.load(f)
    else:
        with io.open(path, "rt", encoding=encoding) as f:
            line = f.read()
            dct = json.loads(line)
    return dct


def write_json(path, dct):
    with open(path, "w") as f:
        json.dump(dct, f, indent=4, ensure_ascii=False)


def mkdir(path, newdir=None):
    if newdir is None:
        target = path
    else:
        target = os.path.join(path, newdir)
    if not os.path.exists(target):
        os.makedirs(target)
        print("Created a new directory: %s" % target)


def read_api_key(llm_type, key_path="keys.json"):
    with open(key_path, "r") as f:
        keys = json.load(f)
    return keys.get(llm_type)


def read_prompt_template(name):
    with open(os.path.join("prompt_templates", f"{name}.txt"), "r", encoding="utf-8") as f:
        return f.read().strip()
 

def safe_json_loads(generated_text, fallback=None, list_type=False):
    """
    Parse the report into a JSON object
    """
    # try:
    #     return json.loads(generated_text)
    # except json.JSONDecodeError as e:
    #     cleaned = (
    #         generated_text.strip()
    #         .removeprefix("```json").removesuffix("```")
    #         .strip("` \n")
    #     )
    #     try:
    #         return json.loads(cleaned)
    #     except json.JSONDecodeError as e2:
    #         print("[JSONDecodeError]", e)
    #         print("[Raw Output]", generated_text[:300])
    #         return fallback

    if list_type:
        begin_index = generated_text.find("[")
        end_index = generated_text.rfind("]")
    else:
        begin_index = generated_text.find("{")
        end_index = generated_text.rfind("}")
    if begin_index < 0 or end_index < 0:
        print(f"Failed to parse the generated text into a JSON object: '{generated_text}'")
        return fallback

    json_text = generated_text[begin_index: end_index + 1]

    try:
        json_obj = json.loads(json_text)
    except Exception as e:
        print(f"Failed to parse the generated text into a JSON object: '{json_text}'")
        print(e)
        return fallback

    if list_type:
        if not isinstance(json_obj, list):
            print(f"The parsed JSON object is not a list: '{json_obj}'")
            return fallback
    else:
        if not isinstance(json_obj, dict):
            print(f"The parsed JSON object is not a dictionary: '{json_obj}'")
            return fallback

    return json_obj

 