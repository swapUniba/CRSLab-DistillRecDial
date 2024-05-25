import json
import os
import re

import editdistance
import torch
from transformers import (
    pipeline, AutoModelForCausalLM, AutoTokenizer,
)

from crslab.config import DATASET_PATH
from crslab.model.base import BaseModel

PROMPT_TEMPLATE = (
            "Pretend you are a course curriculum recommender system. "
            "I will give you a conversation between a user and you (a recommender system). "
            "Based on the conversation, you reply me with 50 topic recommendations without extra sentences."
            "\n\nHere is the conversation: \n'{}' "
            "\n\nList of 50 recommendations: "
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


def match_topics(all_items, predicted_items, threshold=2):
    if not all_items:
        return []

    matched_predictions = []

    for item in predicted_items:
        closest_topic, min_distance = find_similar_topic(item, all_items)

        if min_distance <= threshold:
            matched_predictions.append(closest_topic)
        # TODO: We should not add it if it does not match?
        else:
            matched_predictions.append("OOD")

    return matched_predictions


class HuggingfaceModel(BaseModel):
    def __init__(self, opt, device, dpath=None, resource=None):
        # dataset = opt['dataset']
        # dpath = os.path.join(MODEL_PATH, "kgsf", dataset)
        # resource = resources[dataset]
        dataset = opt['dataset'].lower()
        self.dataset_path = os.path.join(DATASET_PATH, dataset, "none")

        super().__init__(opt, device)

        with open(f"{self.dataset_path}/entity2id.json", 'r', encoding="utf-8") as f:
            self.entity2id = json.load(f)

        self.entity2id["OOD"] = -1

        self.id2entity = {idx: entity for entity, idx in self.entity2id.items()}
        self.all_items = list(self.id2entity.values())

        self.model_id = opt["model_id"]
        self.max_target_length = opt["max_target_length"]
        self.bf16 = opt["bf16"]

        if self.bf16:
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float32

        print(f"Using {self.torch_dtype=}")

        # self.pipe = pipeline(
        #     "text-generation",
        #     model=self.model_id,
        #     model_kwargs={"torch_dtype": self.torch_dtype},
        #     device=self.device,
        # )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            attn_implementation="sdpa",
            torch_dtype=self.torch_dtype,
        )
        self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"



        self.generation_kwargs = dict(
            pad_token_id=self.tokenizer.eos_token_id,  # Ensure padding is consistent
            max_new_tokens=self.max_target_length,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )


        if self.model_id in [
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "eirsteir/crs-llama-3-8b-instruct",
        ]:
            self.generation_kwargs["eos_token_id"] = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ]

    def build_model(self):
        pass


    def recommend(self, batch, mode="test"):

        def format_input(context):
            prompt = PROMPT_TEMPLATE.format(context)
            prompt = self.format_prompt_for_chat(prompt)
            return prompt

        print(batch)
        prompts = [format_input(context) for context in batch["context"]]
        encodings = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**encodings, **self.generation_kwargs)

        outputs = [
            output[len(encodings["input_ids"][i]):]
            for i, output in enumerate(outputs)
        ]

        texts = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )

        all_predicted_recs = [parse_topics(text) for text in texts]
        all_predicted_recs = [match_topics(all_items=self.all_items, predicted_items=recs) for recs in all_predicted_recs]

        all_predicted_recs = [[self.entity2id[rec] for rec in recs] for recs in all_predicted_recs]

        return all_predicted_recs


    def format_prompt_for_chat(self, prompt):
        messages = [{"role": "user", "content": prompt}]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        return prompt

    def converse(self, batch, mode):
        all_responses = []

        for example in batch:
            context = example["context"]  # TODO: fix

            conv = self.tokenizer.apply_chat_template(
                context, add_generation_prompt=True, tokenize=False
            )
            outputs = self.pipe(conv, **self.generation_kwargs)
            response_text = outputs[0]["generated_text"][len(context):]

            all_responses.append({
                "dialog_id": example["dialog_id"],
                "response": response_text
            })

        return all_responses

    def guide(self, batch, mode):
        pass
