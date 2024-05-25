import json
import re

import editdistance
import torch
from transformers import (
    pipeline,
)

from crslab.model.base import BaseModel

PROMPT_TEMPLATE = (
            "Pretend you are a course curriculum recommender system. "
            "I will give you a conversation between a user and you (a recommender system). "
            "Based on the conversation, you reply me with 20 topic recommendations without extra sentences."
            "\n\nHere is the conversation: \n'{}' "
            "\n\nList of 20 recommendations: "
)


def parse_topics(input_string):
    # Define a regex pattern to extract text after the number and period
    pattern = re.compile(r"\d+\.\s+(.*)")

    # Find all matches in the input string and return them as a list
    topics = pattern.findall(input_string)

    return [topic.lower() for topic in topics]




def compute_edit_distance(str1, str2):
    return editdistance.eval(str1, str2)


def find_similar_topic(input_topic, topics):
    # Find the topic in the list that has the smallest edit distance to the input topic
    min_distance = float("inf")
    closest_topic = None

    for topic in topics:
        distance = compute_edit_distance(input_topic, topic)
        if distance < min_distance:
            min_distance = distance
            closest_topic = topic

    return closest_topic, min_distance


def match_topics(true_labels, predicted_labels, threshold=2):
    matched_predictions = []

    for item in predicted_labels:
        closest_topic, min_distance = find_similar_topic(item, true_labels)

        if min_distance <= threshold:
            matched_predictions.append(closest_topic)
        else:
            matched_predictions.append(item)

    return matched_predictions


class HuggingfaceModel(BaseModel):
    def __init__(self, opt, device, dpath=None, resource=None):
        super().__init__(opt, device, dpath, resource)

        with open(f"{self.dataset_path}/entity2id.json", 'r', encoding="utf-8") as f:
            self.entity2id = json.load(f)

        self.model_id = opt["model_id"]
        self.bf16 = opt["bf16"]

        if self.bf16:
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float32  # Need A100 for torch.bfloat16

        print(f"Using {self.torch_dtype=}")

        self.generation_kwargs = dict(
            max_new_tokens=256,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={"torch_dtype": self.torch_dtype},
            device=self.device,
        )

        if self.model_id in [
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "eirsteir/crs-llama-3-8b-instruct",
        ]:
            self.generation_kwargs["eos_token_id"] = [
                self.pipe.tokenizer.eos_token_id,
                self.pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ]

    def build_model(self):
        pass

    def recommend(self, batch, mode="test"):
        all_predicted_recs = []

        for example in batch:
            context = example["context"]
            prompt = PROMPT_TEMPLATE.format(context)
            prompt = self.format_prompt_for_chat(prompt)

            outputs = self.pipe(prompt, **self.generation_kwargs)
            text = outputs[0]["generated_text"][len(prompt):]

            gt_labels = example["item"]
            pred_labels = parse_topics(text)
            pred_labels = match_topics(true_labels=gt_labels, predicted_labels=pred_labels)
            pred_labels = [self.entity2id[label] for label in pred_labels]

            all_predicted_recs.extend(pred_labels)

        return all_predicted_recs


    def format_prompt_for_chat(self, prompt):
        messages = [{"role": "user", "content": prompt}]

        prompt = self.pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        return prompt

    def converse(self, batch, mode):
        return

    def guide(self, batch, mode):
        pass
