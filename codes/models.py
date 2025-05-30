import base64
from PIL import Image
import os

from openai import OpenAI
import google.generativeai as genai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential
)

import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
from qwen_vl_utils import process_vision_info
import torch

import utils


class OpenAIMultimodalModel:
    def __init__(self, model_name):
        self.model_name = model_name

        self.client = OpenAI(api_key=utils.read_api_key("openai"))

    def load_frames(self, path_frames, base_filename):
        filenames = os.listdir(path_frames)
        filenames = [n for n in filenames if n.startswith(base_filename + ".frame_")]
        filenames = sorted(filenames, key=lambda n: int(n.split("_")[-1].split(".")[0]))

        frames = []
        for filename in filenames:
            full_path = os.path.join(path_frames, filename)
            with open(full_path, "rb") as f:
                frame = base64.b64encode(f.read()).decode("utf-8")
                frames.append(frame)
        return frames

    def generate(self, prompt, path_frames, base_filename):
        frames = self.load_frames(path_frames=path_frames, base_filename=base_filename)
        response = self.client.chat.completions.create(
           model=self.model_name,
           messages=[
               {
                   "role": "system",
                   "content": "You are a helpful assistant."
               },
               {
                   "role": "user",
                   "content": [
                       {
                           "type": "text",
                           "text": prompt
                       },
                       *map(
                           lambda x: {
                               "type": "image_url",
                               "image_url": {
                                   "url": f"data:image/jpg;base64,{x}",
                                   "detail": "low"
                               }
                           },
                           frames
                       )
                   ]
               }
           ],
           temperature=0
        )
        generated_text = response.choices[0].message.content
        return generated_text


class GeminiMultimodalModel:
    def __init__(self, model_name):
        self.model_name = model_name

        genai.configure(api_key=utils.read_api_key("gemini"))
        self.client = genai.GenerativeModel(model_name)

    def load_frames(self, path_frames, base_filename):
        filenames = os.listdir(path_frames)
        filenames = [n for n in filenames if n.startswith(base_filename + ".frame_")]
        filenames = sorted(filenames, key=lambda n: int(n.split("_")[-1].split(".")[0]))

        frames = []
        for filename in filenames:
            full_path = os.path.join(path_frames, filename)
            frame = Image.open(full_path)
            frames.append(frame)
        return frames
 
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
    def generate(self, prompt, path_frames, base_filename):
        frames = self.load_frames(path_frames=path_frames, base_filename=base_filename)
        response = self.client.generate_content(
            [prompt] + frames,
            generation_config={"temperature": 0}
        )
        generated_text = response.text
        return generated_text


class QwenMultimodalModel:
    def __init__(self, model_name, max_frames=8):
        self.model_name = model_name
        self.max_frames = max_frames

        self.llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2", # removed due to error in my environment
            device_map="auto",
            trust_remote_code=True
        )

        self.min_pixels = 256*28*28
        # self.max_pixels = 1280*28*28
        self.max_pixels = 512*28*28

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels
        )

    def load_frame_paths(self, path_frames, base_filename):
        filenames = os.listdir(path_frames)
        filenames = [n for n in filenames if n.startswith(base_filename + ".frame_")]
        filenames = sorted(filenames, key=lambda n: int(n.split("_")[-1].split(".")[0]))
        filepaths = [os.path.join(path_frames, n) for n in filenames]
        filepaths = filepaths[-self.max_frames:]
        return filepaths

    def generate(self, prompt, path_frames, base_filename):
        frame_paths = self.load_frame_paths(
            path_frames=path_frames,
            base_filename=base_filename
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frame_paths
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True
        )
        video_kwargs["fps"] = 1.0 # TODO: Should align with the data preprocessing

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs
        )
        inputs = inputs.to(self.llm.device)

        generated_ids = self.llm.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        generated_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return generated_text


class LlavaMultimodalModel:
    def __init__(self, model_name, max_frames=8):
        self.model_name = model_name
        self.max_frames = max_frames

        self.llm = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        self.processor = LlavaNextVideoProcessor.from_pretrained(model_name)

    def load_frames(self, path_frames, base_filename):
        filenames = os.listdir(path_frames)
        filenames = [n for n in filenames if n.startswith(base_filename + ".frame_")]
        filenames = sorted(filenames, key=lambda n: int(n.split("_")[-1].split(".")[0]))

        frames = []
        for filename in filenames:
            full_path = os.path.join(path_frames, filename)
            frame = Image.open(full_path)
            frames.append(frame)

        frames = np.stack([np.array(f.convert("RGB")) for f in frames])
        return frames

    def generate(self, prompt, path_frames, base_filename):
        frames = self.load_frames(
            path_frames=path_frames,
            base_filename=base_filename
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "video"
                    }
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            # tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=text,
            videos=frames,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.llm.device)

        generated_ids = self.llm.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        generated_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]
        return generated_text


class OpenAIModel:
    def __init__(self, model_name="gpt-4o-mini"):
        self.client = OpenAI(api_key=utils.read_api_key("openai"))
        self.model_name = model_name

    def generate(self, prompt):        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )
        generated_text = response.choices[0].message.content
        return generated_text


