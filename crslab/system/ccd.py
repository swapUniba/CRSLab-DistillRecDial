from peft import LoraConfig
from transformers import TrainingArguments
from transformers import (
    set_seed,
)
from loguru import logger

from crslab.evaluator.metrics.gen import PPLMetric
from crslab.system.base import BaseSystem


class HuggingfaceSystem(BaseSystem):
    def __init__(self, opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data, restore_system=False,
                 interact=False, debug=False, tensorboard=False):
        super(HuggingfaceSystem, self).__init__(opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data,
                                                restore_system, interact, debug, tensorboard)


        self.batch_size = opt["batch_size"]

    def step(self, batch, stage, mode):
        if mode != "test":
            return

        if stage == "rec":
            rec_predict = self.model.recommend(batch, mode)
            self.rec_evaluate(rec_predict, batch["item"])

        elif stage == "conv":
            loss, conv_predict = self.model.converse(batch, mode)
            self.conv_evaluate(conv_predict, batch["response"])
            self.evaluator.gen_metrics.add('ppl', PPLMetric(loss))

    def fit(self):
        self.test_recommender()
        self.test_conversation()

    def rec_evaluate(self, rec_predict, item_label):
        for pred, label in zip(rec_predict, item_label):
            self.evaluator.rec_evaluate(pred, label)

    def conv_evaluate(self, conv_predict, conv_label):
        for pred, label in zip(conv_predict, conv_label):
            self.evaluator.gen_evaluate(pred, [label])

    def interact(self):
        pass

    def test_recommender(self):
        logger.info('[Test Recommendation]')

        self.evaluator.reset_metrics()

        for batch in self.test_dataloader.get_rec_data(batch_size=self.batch_size, shuffle=False):
            self.step(batch, stage='rec', mode='test')

        self.evaluator.report(mode='test')

    def test_conversation(self):
        logger.info('[Test Conversation]')

        self.evaluator.reset_metrics()

        for batch in self.test_dataloader.get_conv_data(batch_size=self.batch_size, shuffle=False):
            self.step(batch, stage='conv', mode='test')

        self.evaluator.report(mode='test')


