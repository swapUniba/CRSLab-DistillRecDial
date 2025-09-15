# @Time   : 2020/12/4
# @Author : Chenzhan Shang
# @Email  : czshang@outlook.com

# UPDATE
# @Time   : 2021/1/3
# @Author : Xiaolei Wang
# @email  : wxl1999@foxmail.com
import os

import torch
from loguru import logger

from crslab.config import SAVE_PATH
from crslab.data import dataset_language_map
from crslab.evaluator.metrics.base import AverageMetric
from crslab.evaluator.metrics.gen import PPLMetric
from crslab.system.base import BaseSystem
from crslab.system.utils.functions import ind2txt


class ReDialSystem(BaseSystem):
    """This is the system for KGSF model"""

    def __init__(self, opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data, restore_system=False,
                 interact=False, debug=False, tensorboard=False):
        """

        Args:
            opt (dict): Indicating the hyper parameters.
            train_dataloader (BaseDataLoader): Indicating the train dataloader of corresponding dataset.
            valid_dataloader (BaseDataLoader): Indicating the valid dataloader of corresponding dataset.
            test_dataloader (BaseDataLoader): Indicating the test dataloader of corresponding dataset.
            vocab (dict): Indicating the vocabulary.
            side_data (dict): Indicating the side data.
            restore_system (bool, optional): Indicating if we store system after training. Defaults to False.
            interact (bool, optional): Indicating if we interact with system. Defaults to False.
            debug (bool, optional): Indicating if we train in debug mode. Defaults to False.
            tensorboard (bool, optional) Indicating if we monitor the training performance in tensorboard. Defaults to False. 

        """
        super(ReDialSystem, self).__init__(opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data,
                                           restore_system, interact, debug, tensorboard)
        self.ind2tok = vocab['conv']['ind2tok']
        self.end_token_idx = vocab['conv']['end']
        self.item_ids = side_data['rec']['item_entity_ids']
        self.id2entity = vocab['rec']['id2entity']

        self.rec_optim_opt = opt['rec']
        self.conv_optim_opt = opt['conv']
        self.get_item_name = train_dataloader['rec'].get_item_name
        self.rec_epoch = self.rec_optim_opt['epoch']
        self.conv_epoch = self.conv_optim_opt['epoch']
        self.rec_batch_size = self.rec_optim_opt['batch_size']
        self.conv_batch_size = self.conv_optim_opt['batch_size']

        self.language = dataset_language_map[self.opt['dataset']]
        self.model_name = self.opt['model_name']
        self.predicted_target_senteces = []
        self.predicted_target_recommendations = []

    def rec_evaluate(self, rec_predict, item_label, save=False):
        rec_predict = rec_predict.cpu()
        rec_predict_new = rec_predict.new_full(rec_predict.size(), float('-inf'))
        rec_predict_new[:, self.item_ids] = rec_predict[:, self.item_ids]
        _, rec_ranks = torch.topk(rec_predict_new, 100, dim=-1)
        rec_ranks = rec_ranks.tolist()
        item_label = item_label.tolist()
        items_labels = self.get_item_name(item_label)
        items_ranks = [self.get_item_name(x) for x in rec_ranks]
        for rec_rank, item in zip(rec_ranks, item_label):
            item = self.item_ids.index(item)
            self.evaluator.rec_evaluate(rec_rank, item)
        if save:
            for rec_rank, item in zip(items_ranks, items_labels):
                self.predicted_target_recommendations.append({
                    'target': item,
                    'predicted': rec_rank,
                })

    def save_response_file(self):
        import json
        os.makedirs(os.path.join(SAVE_PATH, self.model_name), exist_ok=True)
        with open(os.path.join(SAVE_PATH, self.model_name, 'predicted_target_senteces.json'), 'w', encoding='utf-8') as f:
            json.dump(self.predicted_target_senteces, f, ensure_ascii=False, indent=4)

    def save_recommendation_file(self):
        import json
        os.makedirs(os.path.join(SAVE_PATH, self.model_name), exist_ok=True)
        with open(os.path.join(SAVE_PATH, self.model_name, 'predicted_target_recommendations.json'), 'w', encoding='utf-8') as f:
            json.dump(self.predicted_target_recommendations, f, ensure_ascii=False, indent=4)

    def conv_evaluate(self, prediction, response, save=False):
        prediction = prediction.tolist()
        response = response.tolist()
        for p, r in zip(prediction, response):
            p_str = ind2txt(p, self.ind2tok, self.end_token_idx)
            r_str = ind2txt(r, self.ind2tok, self.end_token_idx)
            self.evaluator.gen_evaluate(p_str, [r_str])
            if save:
                self.predicted_target_senteces.append({
                    'target': r_str,
                    'predicted': p_str,
                })

    def step(self, batch, stage, mode):
        assert stage in ('rec', 'conv')
        assert mode in ('train', 'valid', 'test')

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)

        if stage == 'rec':
            rec_loss, rec_scores = self.rec_model.forward(batch, mode=mode)
            rec_loss = rec_loss.sum()
            if mode == 'train':
                self.backward(rec_loss)
            else:
                self.rec_evaluate(rec_scores, batch['item'], save=mode=='test')
            rec_loss = rec_loss.item()
            self.evaluator.optim_metrics.add("rec_loss", AverageMetric(rec_loss))
        else:
            gen_loss, preds = self.conv_model.forward(batch, mode=mode)
            gen_loss = gen_loss.sum()
            if mode == 'train':
                self.backward(gen_loss)
            else:
                self.conv_evaluate(preds, batch['response'], save=mode=='test')
            gen_loss = gen_loss.item()
            self.evaluator.optim_metrics.add('gen_loss', AverageMetric(gen_loss))
            self.evaluator.gen_metrics.add('ppl', PPLMetric(gen_loss))

    def train_recommender(self):
        self.init_optim(self.rec_optim_opt, self.rec_model.parameters())

        for epoch in range(self.rec_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Recommendation epoch {str(epoch)}]')
            logger.info('[Train]')
            for batch in self.train_dataloader['rec'].get_rec_data(batch_size=self.rec_batch_size):
                self.step(batch, stage='rec', mode='train')
            self.evaluator.report(epoch=epoch, mode='train')  # report train loss
            # val
            logger.info('[Valid]')
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader['rec'].get_rec_data(batch_size=self.rec_batch_size, shuffle=False):
                    self.step(batch, stage='rec', mode='valid')
                self.evaluator.report(epoch=epoch, mode='valid')  # report valid loss
                # early stop
                metric = self.evaluator.optim_metrics['rec_loss']
                if self.early_stop(metric):
                    break
        # test
        logger.info('[Test]')
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader['rec'].get_rec_data(batch_size=self.rec_batch_size, shuffle=False):
                self.step(batch, stage='rec', mode='test')
            self.evaluator.report(mode='test')
        # save
        self.save_recommendation_file()

    def train_conversation(self):
        self.init_optim(self.conv_optim_opt, self.conv_model.parameters())

        for epoch in range(self.conv_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Conversation epoch {str(epoch)}]')
            logger.info('[Train]')
            for batch in self.train_dataloader['conv'].get_conv_data(batch_size=self.conv_batch_size):
                self.step(batch, stage='conv', mode='train')
            self.evaluator.report(epoch=epoch, mode='train')
            # val
            logger.info('[Valid]')
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader['conv'].get_conv_data(batch_size=self.conv_batch_size,
                                                                         shuffle=False):
                    self.step(batch, stage='conv', mode='valid')
                self.evaluator.report(epoch=epoch, mode='valid')
                metric = self.evaluator.optim_metrics['gen_loss']
                if self.early_stop(metric):
                    break
        # test
        logger.info('[Test]')
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader['conv'].get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                self.step(batch, stage='conv', mode='test')
            self.evaluator.report(mode='test')
        # save
        self.save_response_file()

    def fit(self):
        self.train_recommender()
        self.train_conversation()

    def interact(self):
        pass
