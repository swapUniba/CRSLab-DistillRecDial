

import json
import os
from copy import copy

from loguru import logger
from tqdm import tqdm

from crslab.config import DATASET_PATH
from crslab.data.dataset.base import BaseDataset
from .resources import resources


class CCDataset(BaseDataset):
    """

    Attributes:
        train_data: train dataset.
        valid_data: valid dataset.
        test_data: test dataset.
        vocab (dict): ::

            {
                'tok2ind': map from token to index,
                'ind2tok': map from index to token,
                'entity2id': map from entity to index,
                'id2entity': map from index to entity,
                'word2id': map from word to index,
                'vocab_size': len(self.tok2ind),
                'n_entity': max(self.entity2id.values()) + 1,
                'n_word': max(self.word2id.values()) + 1,
            }

    Notes:
        ``'unk'`` must be specified in ``'special_token_idx'`` in ``resources.py``.

    """

    def __init__(self, opt, tokenize, restore=False, save=False):
        """Specify tokenized resource and init base dataset.

        Args:
            opt (Config or dict): config for dataset or the whole system.
            tokenize (str): how to tokenize dataset.
            restore (bool): whether to restore saved dataset which has been processed. Defaults to False.
            save (bool): whether to save dataset after processing. Defaults to False.

        """
        resource = resources[tokenize]
        self.special_token_idx = resource['special_token_idx']
        self.unk_token_idx = self.special_token_idx['unk']
        self.format_for_redial = opt.get("format_for_redial", False)
        self.keep_text = opt.get("keep_text", False)

        dpath = os.path.join(DATASET_PATH, "ccd", tokenize)
        super().__init__(opt, dpath, resource, restore, save)


    def _load_data(self):
        train_data, valid_data, test_data = self._load_raw_data()
        self._load_vocab()
        self._load_other_data()

        vocab = {
            'tok2ind': self.tok2ind,
            'ind2tok': self.ind2tok,
            'entity2id': self.entity2id,
            'id2entity': self.id2entity,
            'word2id': self.word2id,
            'vocab_size': len(self.tok2ind),
            'n_entity': self.n_entity,
            'n_word': self.n_word,
        }
        vocab.update(self.special_token_idx)

        return train_data, valid_data, test_data, vocab

    def _load_raw_data(self):
        # load train/valid/test data
        with open(os.path.join(self.dpath, 'train_data.json'), 'r', encoding='utf-8') as f:
            train_data = json.load(f)
            logger.debug(f"[Load train data from {os.path.join(self.dpath, 'train_data.json')}]")
        with open(os.path.join(self.dpath, 'valid_data.json'), 'r', encoding='utf-8') as f:
            valid_data = json.load(f)
            logger.debug(f"[Load valid data from {os.path.join(self.dpath, 'valid_data.json')}]")
        with open(os.path.join(self.dpath, 'test_data.json'), 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            logger.debug(f"[Load test data from {os.path.join(self.dpath, 'test_data.json')}]")

        return train_data, valid_data, test_data

    def _load_vocab(self):
        self.tok2ind = json.load(open(os.path.join(self.dpath, 'token2id.json'), 'r', encoding='utf-8'))
        self.ind2tok = {idx: word for word, idx in self.tok2ind.items()}

        logger.debug(f"[Load vocab from {os.path.join(self.dpath, 'token2id.json')}]")
        logger.debug(f"[The size of token2index dictionary is {len(self.tok2ind)}]")
        logger.debug(f"[The size of index2token dictionary is {len(self.ind2tok)}]")

    def _load_other_data(self):
        self.entity2id = json.load(
            open(os.path.join(self.dpath, 'entity2id.json'), 'r', encoding='utf-8'))  # {entity: entity_id}
        self.id2entity = {idx: entity for entity, idx in self.entity2id.items()}
        self.n_entity = max(self.entity2id.values()) + 1

        self.word2id = json.load(open(os.path.join(self.dpath, 'token2id.json'), 'r', encoding='utf-8'))
        self.n_word = max(self.word2id.values()) + 1

    def _data_preprocess(self, train_data, valid_data, test_data):
        processed_train_data = self._raw_data_process(train_data)
        logger.debug("[Finish train data process]")
        processed_valid_data = self._raw_data_process(valid_data)
        logger.debug("[Finish valid data process]")
        processed_test_data = self._raw_data_process(test_data)
        logger.debug("[Finish test data process]")
        processed_side_data = self._side_data_process()
        logger.debug("[Finish side data process]")
        return processed_train_data, processed_valid_data, processed_test_data, processed_side_data

    def _raw_data_process(self, raw_data):
        return self._raw_data_process_contextual(raw_data)

    def _raw_data_process_contextual(self, raw_data):
        augmented_convs = [self._convert_tokens_and_words_to_ids(conversation) for conversation in
                           tqdm(raw_data)]
        augmented_conv_dicts = []
        # This puts it in the correct format
        for conv in tqdm(augmented_convs):
            augmented_conv_dicts.extend(self._augment_and_add(conv))
        return augmented_conv_dicts

    def _convert_tokens_and_words_to_ids(self, conversation):
        augmented_messages = []

        for utt in conversation["messages"]:
            text_tokens_ids, word_ids = utt["text"], utt["word"]

            if self.format_for_redial:
                text_tokens_ids, word_ids = utt["annotated_text"], utt["annotated_word"]

            if not self.keep_text:
                text_tokens_ids = [self.tok2ind.get(word, self.unk_token_idx) for word in text_tokens_ids]
                word_ids = [self.word2id[word] for word in word_ids if word in self.word2id]

            role = "Seeker" if utt["role"] == "user" else "Recommender"

            augmented_messages.append({
                "role": role,
                "text": text_tokens_ids,
                "item": utt["item_ids"],
                "entity": utt["entity_ids"],
                "word": word_ids
            })

        return augmented_messages

    def _augment_and_add(self, raw_conv_dict):
        """Builds conversation history (context) for a single conversation."""
        augmented_conv_dicts = []
        context_messages, context_tokens, context_entities, context_words, context_items = [], [], [], [], []
        entity_set, word_set = set(), set()

        for i, utt in enumerate(raw_conv_dict):
            text_tokens, entities, items, words = utt["text"], utt["entity"], utt["item"], utt["word"]

            if len(context_tokens) > 0:
                conv_dict = {
                    "role": utt["role"],
                    "context_tokens": copy(context_tokens),
                    "response": text_tokens,
                    "context_messages": copy(context_messages),
                    "context_entities": copy(context_entities),
                    "context_words": copy(context_words),
                    "context_items": copy(context_items),
                    "items": items,
                }
                augmented_conv_dicts.append(conv_dict)

            context_messages.append({"role": utt["role"], "content": text_tokens})
            context_tokens.append(text_tokens)
            context_items += items

            for entity in entities + items:
                if entity not in entity_set:
                    entity_set.add(entity)
                    context_entities.append(entity)
            for word in words:
                if word not in word_set:
                    word_set.add(word)
                    context_words.append(word)

        return augmented_conv_dicts


    def _side_data_process(self):
        item_entity_ids = json.load(open(os.path.join(self.dpath, 'item_ids.json'), 'r', encoding='utf-8'))
        logger.debug('[Load topic entity ids]')

        side_data = {
            "entity_kg": {},
            "word_kg": {},
            "item_entity_ids": item_entity_ids,
        }
        return side_data

