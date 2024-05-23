

import json
import os
from collections import defaultdict
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
        # dbpedia
        self.entity2id = json.load(
            open(os.path.join(self.dpath, 'entity2id.json'), 'r', encoding='utf-8'))  # {entity: entity_id}
        self.id2entity = {idx: entity for entity, idx in self.entity2id.items()}
        self.n_entity = max(self.entity2id.values()) + 1
        # {head_entity_id: [(relation_id, tail_entity_id)]}
        # self.entity_kg = json.load(open(os.path.join(self.dpath, 'dbpedia_subkg.json'), 'r', encoding='utf-8'))
        # logger.debug(
        #     f"[Load entity dictionary and KG from {os.path.join(self.dpath, 'entity2id.json')} and {os.path.join(self.dpath, 'dbpedia_subkg.json')}]")

        # TODO: do we need kgs? Can we use the same as original Redial?
        # conceptNet
        # {concept: concept_id}
        # self.word2id = json.load(open(os.path.join(self.dpath, 'concept2id.json'), 'r', encoding='utf-8'))
        # self.n_word = max(self.word2id.values()) + 1
        # -----
        self.word2id = json.load(open(os.path.join(self.dpath, 'token2id.json'), 'r', encoding='utf-8'))
        self.n_word = max(self.word2id.values()) + 1
        # -----

        # # {relation\t concept \t concept}
        # self.word_kg = open(os.path.join(self.dpath, 'conceptnet_subkg.txt'), 'r', encoding='utf-8')
        # logger.debug(
        #     f"[Load word dictionary and KG from {os.path.join(self.dpath, 'concept2id.json')} and {os.path.join(self.dpath, 'conceptnet_subkg.txt')}]")

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
        # This merges consequtive utterances from the same role into one conversation
        # augmented_convs = [self._merge_conv_data(conversation["dialog"]) for conversation in tqdm(raw_data)]
        augmented_conv_dicts = []

        # This puts it in the correct format
        for conv in tqdm(raw_data):
            augmented_conv_dicts.extend(self._augment_and_add(conv))
        return augmented_conv_dicts


    # TODO: consider replacing items in the text with ids for redial '@id' (see tgredial.py #201)
    def _augment_and_add(self, raw_conv_dict):
        """Builds conversation history (context) for a single conversation."""
        augmented_conv_dicts = []
        context_tokens, context_entities, context_words, context_items, contexts = [], [], [], [], []
        entity_set, word_set = set(), set()

        for i, conv in enumerate(raw_conv_dict["messages"]):
            text_tokens, entities, topics, words, content = conv["annotated_text"], conv["entity_ids"], conv["item_ids"], conv["annotated_word"], conv["content"]

            if len(context_tokens) > 0:
                conv_dict = {
                    "role": "Seeker" if conv["role"] == "user" else "Recommender",
                    "context_tokens": copy(context_tokens),
                    "response": text_tokens,
                    "context_entities": copy(context_entities),
                    "context_words": copy(context_words),
                    "context_items": copy(context_items),
                    "items": topics,
                    # "content": content,  # TODO: does this work? yes, see tgredial
                    # "context": copy(contexts),
                }
                augmented_conv_dicts.append(conv_dict)

            context_tokens.append(text_tokens)
            contexts.append(content),
            context_items += topics

            for entity in entities + topics:
                if entity not in entity_set:
                    entity_set.add(entity)
                    context_entities.append(entity)

            for word in words:
                if word not in word_set:
                    word_set.add(word)
                    context_words.append(word)

        return augmented_conv_dicts

    def _side_data_process(self):
        # processed_entity_kg = self._entity_kg_process()
        # logger.debug("[Finish entity KG process]")
        # processed_word_kg = self._word_kg_process()
        # logger.debug("[Finish word KG process]")
        movie_entity_ids = json.load(open(os.path.join(self.dpath, 'item_ids.json'), 'r', encoding='utf-8'))
        logger.debug('[Load topic entity ids]')

        side_data = {
            "entity_kg": [], #processed_entity_kg,
            "word_kg": [], #processed_word_kg,
            "item_entity_ids": movie_entity_ids,
        }
        return side_data
