import os
import time
import random
import numpy as np
import argparse

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from transformers import Trainer, TrainingArguments
from transformers import BertModel, BertPreTrainedModel, RobertaModel, RobertaConfig
from torch.nn import CrossEntropyLoss
from unpooled_dataset import UnpooledDataset, variable_collate_fn, ModelType
from transformers import RobertaTokenizer, BertTokenizer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() > 0:
        torch.cuda.manual_seed_all(seed)

def extract_embedding(token_embedding, start_idx_mask,
                      head_idx, pp_idx, children_idx,
                      batch_size):
    # get start idx embedding of each tokens
    start_idx_lens = start_idx_mask.sum(1).view(-1)
    embd_selected = torch.masked_select(token_embedding, start_idx_mask.unsqueeze(2))\
        .view(-1, token_embedding.size()[-1])
    embd_split = torch.split(embd_selected, start_idx_lens.tolist())
    embd_padded = pad_sequence(embd_split, batch_first=True, padding_value=0)

    # get children and pp embedding
    pp_embd = embd_padded[torch.arange(batch_size), pp_idx]
    children_embd = embd_padded[torch.arange(batch_size), children_idx]

    # extract head tokens
    head_idx_expaded = head_idx.unsqueeze(-1).repeat(1, 1, embd_padded.size()[2])
    head_padded = torch.gather(embd_padded, 1, head_idx_expaded)

    # combined embedding
    pp_embd_expanded = pp_embd.unsqueeze(1).repeat(1, head_padded.size()[1], 1)
    children_embd_expanded = children_embd.unsqueeze(1).repeat(1, head_padded.size()[1], 1)
    combined_embd = torch.cat([head_padded, pp_embd_expanded, children_embd_expanded], 2)
    return combined_embd

def get_loss(logits, n_heads, batch_size, labels):
    # get head_mask
    batch_max_heads = torch.max(n_heads)
    range_tensor = n_heads.new_ones(batch_size, batch_max_heads).cumsum(dim=1)
    label_mask = (range_tensor <= n_heads.unsqueeze(1)).long()

    logits = logits + (label_mask + 1e-45).log()
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(logits, labels)

    best_scores, pred = logits.max(dim=1)
    return (loss, pred)

class AttachmentModelBert(BertPreTrainedModel):
    def __init__(self, config):
        super(AttachmentModelBert, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.hidden = nn.Linear(3 * config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(4 * config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids, attention_mask,
                head_idx, start_idx_mask, pp_idx, children_idx,
                n_heads, labels):
        batch_size = input_ids.size(0)
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True
        )
        token_embedding = self.dropout(outputs.last_hidden_state)
        combined_embd = extract_embedding(token_embedding, start_idx_mask,
                      head_idx, pp_idx, children_idx,
                      batch_size)
        logits = self.hidden(combined_embd)
        logits = torch.tanh(logits)
        #skip connection
        logits = torch.cat([combined_embd, logits], 2)
        logits = self.classifier(logits).squeeze(-1)
        return get_loss(logits, n_heads, batch_size, labels)

class AttachmentModelRoberta(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"
    def __init__(self, config):
        super(AttachmentModelRoberta, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.hidden = nn.Linear(3 * config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(4 * config.hidden_size, 1)
        self.init_weights()

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
        combined_embd = extract_embedding(token_embedding, start_idx_mask,
                      head_idx, pp_idx, children_idx,
                      batch_size)
        logits = self.hidden(combined_embd)
        logits = torch.tanh(logits)
        #skip connection
        logits = torch.cat([combined_embd, logits], 2)
        logits = self.classifier(logits).squeeze(-1)
        return get_loss(logits, n_heads, batch_size, labels)

def train(datasets, modeloutput, model):
    train_dataset, eval_dataset, test_dataset = datasets
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameter: {n_params}')
    training_args = TrainingArguments(
        output_dir=f'{modeloutput}/results_{int(time.time())}',  # output directory
        num_train_epochs=3,  # total # of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir=f'{modeloutput}/logs',  # directory for storing logs
        save_total_limit=1,
        seed=42
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
    print('Dev set result: ', result)
    if test_dataset is not None:
        result = trainer.evaluate(test_dataset)
        print('Test set result: ', result)

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
    parser.add_argument("-d", dest="devfile", required=True,
                        help="dev file path")
    parser.add_argument("-e", dest="testfile", required=False,
                        default=None,
                        help="test file path")
    parser.add_argument("-o", dest="modeloutput", required=True,
                        help="model output file path")
    parser.add_argument('-m', dest="model_type", required=True,
                        choices=[1, 2],
                        type = int,
                        help='1=Roberta, 2=Bert')
    args = parser.parse_args()

    trainfile = os.path.expanduser(args.trainfile)
    devfile = os.path.expanduser(args.devfile)
    modeloutput = os.path.expanduser(args.modeloutput)
    model_type = [ModelType.ROBERTA, ModelType.BERT][args.model_type-1]

    if model_type == ModelType.ROBERTA:
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = AttachmentModelRoberta.from_pretrained('roberta-base')
    elif model_type == ModelType.BERT:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = AttachmentModelBert.from_pretrained('bert-base-uncased')
    else:
        raise Exception(f'Model type {model_type} not known.')

    train_dataset = UnpooledDataset(trainfile, model_type, tokenizer)
    eval_dataset = UnpooledDataset(devfile, model_type, tokenizer)
    datasets = (train_dataset, eval_dataset)
    test_dataset = None
    if args.testfile is not None:
        testfile = os.path.expanduser(args.testfile)
        test_dataset = UnpooledDataset(testfile, model_type, tokenizer)

    datasets += (test_dataset,)
    train(datasets, modeloutput, model)



