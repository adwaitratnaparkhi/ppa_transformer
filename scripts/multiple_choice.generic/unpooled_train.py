import os
import sys
import time
import random
import numpy as np
import argparse

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from transformers import Trainer, TrainingArguments
from transformers import BertModel, BertPreTrainedModel, RobertaModel, RobertaConfig
from torch.nn import CrossEntropyLoss
from unpooled_dataset import UnpooledDataset, variable_collate_fn

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() > 0:
        torch.cuda.manual_seed_all(seed)

class PPAttachmentBert(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"
    def __init__(self, config):
        super(PPAttachmentBert, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.hidden = nn.Linear(3 * config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()
        self._is_ranking = False

    @property
    def is_ranking(self):
        """I'm the 'x' property."""
        return self._is_ranking

    @is_ranking.setter
    def is_ranking(self, value):
        self._is_ranking = value
        if self._is_ranking:
            print('ranking loss function is used')
        else:
            print('cross entropy loss function is used')

    def forward(self, input_ids, attention_mask,
                head_idx, start_idx_mask, pp_idx, children_idx,
                n_heads, labels):
        batch_size = input_ids.size(0)
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True
        )
        token_embedding = self.dropout(outputs.last_hidden_state)
        #get start idx embedding of each tokens
        start_idx_lens = start_idx_mask.sum(1).view(-1)
        embd_selected = torch.masked_select(token_embedding,
                            start_idx_mask.unsqueeze(2)).view(-1,token_embedding.size()[-1])
        embd_split = torch.split(embd_selected, start_idx_lens.tolist())
        embd_padded = pad_sequence(embd_split, batch_first=True, padding_value=0)

        #get children and pp
        pp_emdb = embd_padded[torch.arange(batch_size), pp_idx]
        children_emdb = embd_padded[torch.arange(batch_size), children_idx]

        #extract head tokens
        head_idx_expaded = head_idx.unsqueeze(-1).repeat(1, 1, embd_padded.size()[2])
        head_padded = torch.gather(embd_padded, 1, head_idx_expaded)

        #combined embedding
        pp_emdb_expanded = pp_emdb.unsqueeze(1).repeat(1, head_padded.size()[1], 1)
        children_emdb_expanded = children_emdb.unsqueeze(1).repeat(1, head_padded.size()[1], 1)
        combined_emdb = torch.cat([head_padded, pp_emdb_expanded, children_emdb_expanded], 2)
        logits = self.hidden(combined_emdb)
        logits = torch.tanh(logits)
        logits = self.classifier(logits).squeeze(-1)

        #get head_mask
        batch_max_heads = torch.max(n_heads)
        range_tensor = n_heads.new_ones(batch_size, batch_max_heads).cumsum(dim=1)
        label_mask = (range_tensor <= n_heads.unsqueeze(1)).long()

        #adding 0 to maked values 1 and -inf to maked values 0
        logits = logits + (label_mask + 1e-45).log()

        ####
        #ranking loss
        if self.is_ranking:
            loss_fun = torch.nn.MultiMarginLoss(reduction = 'none')
            #max(0, (1-(true-other)) is 0 when other is -inf so no gradient flow
            loss = loss_fun(logits, labels)
            loss = (loss*batch_max_heads)/n_heads.float()
            loss = loss.mean()
        else:
            #cross entropy loss
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        best_scores, pred = logits.max(dim=1)
        return (loss, pred)

def train(train_dataset, eval_dataset, is_ranking):
    model = PPAttachmentBert.from_pretrained('roberta-base')
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameter: {n_params}')
    model.is_ranking = is_ranking
    training_args = TrainingArguments(
        output_dir=f'./results_{int(time.time())}',  # output directory
        num_train_epochs=3,  # total # of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='../logs',  # directory for storing logs
        save_total_limit=1,
        seed = 42
    )
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=variable_collate_fn,
    )
    trainer.train()
    trainer.save_model()
    result = trainer.evaluate()
    print(result)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def compute_metrics(p):
    preds = p.predictions
    return {"acc": simple_accuracy(preds, p.label_ids)}

if __name__ == '__main__':
    set_seed(42)

    parser = argparse.ArgumentParser(description='Parsing experiment files')
    parser.add_argument("-t", dest="trainfile", required=True,
                        help="train file path")
    parser.add_argument("-d", dest="testfile", required=True,
                        help="test file path")
    parser.add_argument('-r', dest='ranking_loss', action='store_true',
                        default = False,
                        help="if ranking loss enabled")
    args = parser.parse_args()

    trainfile = os.path.expanduser(args.trainfile)
    testfile = os.path.expanduser(args.testfile)
    is_ranking = args.ranking_loss
    train_dataset = UnpooledDataset(trainfile)
    eval_dataset = UnpooledDataset(testfile)
    train(train_dataset, eval_dataset, is_ranking)



