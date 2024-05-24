from peft import LoraConfig
from transformers import TrainingArguments
from transformers import (
    set_seed,
)
from loguru import logger

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

        batch = [d.to(self.device) for d in batch]

        if stage == "rec":
            rec_loss, rec_predict = self.model.recommend(batch, mode)
            self.rec_evaluate(rec_predict, batch[-1])

        elif stage == "conv":
            conv_loss, conv_predict = self.model.converse(batch, mode)
            self.conv_evaluate(conv_predict, batch[-1])


    def fit(self):
        self.rec_evaluate()
        self.conv_evaluate()

    def rec_evaluate(self):
        logger.info('[Test]')
        for batch in self.test_dataloader.get_rec_data(batch_size=self.batch_size, shuffle=False):
            self.step(batch, stage='conv', mode='test')
        self.evaluator.report(mode='test')

    def conv_evaluate(self):
        logger.info('[Test]')
        for batch in self.test_dataloader.get_conv_data(batch_size=self.batch_size, shuffle=False):
            self.step(batch, stage='conv', mode='test')
        self.evaluator.report(mode='test')

    def interact(self):
        pass

    def conv_evaluate(self):
        pass


