# @Time   : 2021/3/11
# @Author : Beichen Zhang
# @Email  : zhangbeichen724@gmail.com
# Optimized version

import torch
import numpy as np
from tqdm import tqdm
from functools import lru_cache
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

from crslab.data.dataloader.base import BaseDataLoader
from crslab.data.dataloader.utils import add_start_end_token_idx, padded_tensor, truncate, merge_utt


class InspiredDataLoader(BaseDataLoader):
    """Optimized dataloader for model Inspired."""

    def __init__(self, opt, dataset, vocab, get_item_name=None):
        """Initialize the dataloader with optimized configurations."""
        super().__init__(opt, dataset)

        self.n_entity = vocab['n_entity']
        self.pad_token_idx = vocab['pad']
        self.start_token_idx = vocab['start']
        self.end_token_idx = vocab['end']
        self.unk_token_idx = vocab['unk']
        self.conv_bos_id = vocab['start']
        self.cls_id = vocab['start']
        self.sep_id = vocab['end']
        self.sent_split_idx = vocab.get('sent_split', vocab['end'])

        self.pad_entity_idx = vocab['pad_entity']
        self.pad_word_idx = vocab['pad_word']

        self.tok2ind = vocab['tok2ind']
        self.ind2tok = vocab['ind2tok']
        self.id2entity = vocab['id2entity']

        self.context_truncate = opt.get('context_truncate', None)
        self.response_truncate = opt.get('response_truncate', None)
        self.get_item_name = get_item_name
        
        # Cache for processed contexts
        self._context_cache = {}
        
        # Pre-compute constants used repeatedly
        self._constants = {
            'start_end': [self.start_token_idx, self.end_token_idx],
            'sent_split': [self.sent_split_idx]
        }

    def rec_process_fn(self, *args, **kwargs):
        """Process recommendation data with optimized approach."""
        augment_dataset = []
        recommender_convs = [conv for conv in self.dataset if conv['role'] == 'Recommender']
        
        for conv_dict in tqdm(recommender_convs):
            # Base dictionary with common fields
            base_dict = {
                'role': conv_dict['role'],
                'context_tokens': conv_dict['context_tokens'],
                # Add other necessary fields without items
            }
            
            # For each movie, create a shallow dictionary with only necessary changes
            for movie in conv_dict['items']:
                # Create a new dict with only the necessary fields
                aug_dict = base_dict.copy()  # Shallow copy of base dict
                aug_dict['item'] = movie
                # Only copy other fields if they're different from base
                for key, value in conv_dict.items():
                    if key not in base_dict and key != 'items':
                        aug_dict[key] = value
                        
                augment_dataset.append(aug_dict)
                
        return augment_dataset

    @lru_cache(maxsize=1024)
    def _process_rec_context_cached(self, context_tuple):
        """Process and cache context processing for reuse.
        
        The input is a tuple (for hashability) but converted to list inside.
        """
        # Convert tuple of tuples to list of lists
        context_tokens = [[token for token in utterance] for utterance in context_tuple]
        return self._process_rec_context(context_tokens)
         
    def _process_rec_context(self, context_tokens):
        """Process recommendation context with optimizations."""
        # Pre-allocate the new list
        compact_context = []
        
        # First item doesn't need special handling
        if context_tokens and len(context_tokens) > 0:
            compact_context.append(context_tokens[0])
            
            # Process remaining items with sent_split prepended
            for utterance in context_tokens[1:]:
                # Create new list with sent_split prepended
                new_utterance = [self.sent_split_idx] + utterance
                compact_context.append(new_utterance)
        
        # Merge utterances and apply truncation in one step if possible
        merged = merge_utt(compact_context)
        truncated = truncate(merged, self.context_truncate - 2, truncate_tail=False)
        
        # Add start/end tokens
        result = [self.start_token_idx] + truncated + [self.end_token_idx]
        
        return result

    def rec_batchify(self, batch):
        """Process recommendation batches efficiently."""
        batch_context = []
        batch_movie_id = []

        # Process contexts in parallel for large batches
        if len(batch) > 100:
            # Convert to hashable type for cache
            context_tuples = [tuple(tuple(u) for u in conv_dict['context_tokens']) for conv_dict in batch]
            
            # Process in parallel
            with ThreadPoolExecutor() as executor:
                batch_context = list(executor.map(self._process_rec_context_cached, context_tuples))
        else:
            # Serial processing for small batches
            for conv_dict in batch:
                context = self._process_rec_context(conv_dict['context_tokens'])
                batch_context.append(context)

        # Extract movie IDs (fast operation)
        batch_movie_id = [conv_dict['item'] for conv_dict in batch]

        # Convert to tensors
        batch_context_tensor = padded_tensor(batch_context,
                                          self.pad_token_idx,
                                          max_len=self.context_truncate)
        batch_mask = (batch_context_tensor != self.pad_token_idx).long()

        return (batch_context_tensor, batch_mask, torch.tensor(batch_movie_id))

    def conv_batchify(self, batch):
        """Process conversation batches efficiently."""
        batch_size = len(batch)
        batch_roles = np.zeros(batch_size, dtype=np.int64)
        batch_context_tokens = []
        batch_response = []

        for i, conv_dict in enumerate(batch):
            # Set role (0 for Seeker, 1 for Recommender)
            if conv_dict['role'] != 'Seeker':
                batch_roles[i] = 1
                
            # Process context tokens
            context_tokens = []
            for j, utter in enumerate(conv_dict['context_tokens']):
                if j < len(conv_dict['context_tokens']) - 1:
                    # Add conv_bos_id to all but the last utterance
                    context_tokens.append(utter + [self.conv_bos_id])
                else:
                    # Last utterance doesn't need the conv_bos_id
                    context_tokens.append(utter)
            
            # Merge, truncate and add to batch
            merged = merge_utt(context_tokens)
            truncated = truncate(merged, max_length=self.context_truncate, truncate_tail=False)
            batch_context_tokens.append(truncated)
            
            # Process response
            truncated_response = truncate(conv_dict['response'], max_length=self.response_truncate - 2)
            response_with_tokens = [self.start_token_idx] + truncated_response + [self.end_token_idx]
            batch_response.append(response_with_tokens)

        # Convert to tensors
        batch_context_tensor = padded_tensor(
            items=batch_context_tokens,
            pad_idx=self.pad_token_idx,
            max_len=self.context_truncate,
            pad_tail=False
        )
        batch_response_tensor = padded_tensor(
            items=batch_response,
            pad_idx=self.pad_token_idx,
            max_len=self.response_truncate,
            pad_tail=True
        )
        
        batch_input_ids = torch.cat((batch_context_tensor, batch_response_tensor), dim=1)
        batch_roles = torch.tensor(batch_roles)

        return (batch_roles,
                batch_input_ids,
                batch_context_tensor,
                batch_response_tensor)

    def policy_batchify(self, batch):
        pass