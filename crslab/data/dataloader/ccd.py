# @Time   : 2023/6/14
# @Author : Xinyu Tang
# @Email  : txy20010310@163.com
from copy import deepcopy

from tqdm import tqdm
from datasets import Dataset as HfDataset
from transformers import AutoTokenizer

from crslab.data.dataloader.base import BaseDataLoader


class HugggingfaceDataLoader(BaseDataLoader):
    """Hf chat dataset dataloader"""

    def __init__(self, opt, dataset, vocab):
        super().__init__(opt, dataset)

        self.id2entity = vocab['id2entity']

    def resolve_role(self, role):
        return "user" if role == "Seeker" else "assistant"

    def batchify(self, batch):
        batch_role = []
        batch_context = []
        batch_movies = []
        batch_response = []

        for conv_dict in batch:
            batch_role.append(conv_dict['role'])
            batch_context.append(conv_dict['context'])
            batch_movies.append(conv_dict['item'])
            batch_response.append(conv_dict['response'])

        return {
            'role': batch_role,
            'context': batch_context,
            'item': batch_movies,
            "response": batch_response,
        }


    def rec_process_fn(self):
        augment_dataset = []
        for conv_dict in tqdm(self.dataset):
            if conv_dict['role'] == 'Recommender':
                for item in conv_dict['items']:
                    augment_conv_dict = deepcopy(conv_dict)
                    augment_conv_dict['item'] = item
                    augment_conv_dict["role"] = self.resolve_role(augment_conv_dict["role"])
                    augment_conv_dict["context"] = " ".join(f"{self.resolve_role(utt['role'])}: {utt['content']}" for utt in augment_conv_dict["context_messages"])
                    augment_dataset.append(augment_conv_dict)
        return augment_dataset

    def rec_batchify(self, batch):
        return self.batchify(batch)

    def conv_process_fn(self):
        augment_dataset = []
        for conv_dict in tqdm(self.dataset):
            role = self.resolve_role(conv_dict["role"])
            context = [{"role": self.resolve_role(utt['role']), "content": utt['content']} for utt in conv_dict["context_messages"]]

            if conv_dict['role'] == 'Recommender':
                augment_conv_dict = {
                    'role': role,
                    'entity': conv_dict['context_entities'],
                    'context': context,
                    "response": conv_dict["response"],
                    'item': conv_dict['items']
                }
                augment_dataset.append(augment_conv_dict)
        return augment_dataset

    def conv_batchify(self, batch):
        return self.batchify(batch)
