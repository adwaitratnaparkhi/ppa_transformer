import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

import json
class ModelType(object):
    ROBERTA = 'roberta'
    BERT = 'bert'

def tokenize_sentence(sentence, model_type, tokenizer):
    if model_type == ModelType.ROBERTA:
        sentence_tokens = [tokenizer.tokenize(sentence[0])] + \
                          [tokenizer.tokenize(f' {t}') for t in sentence[1:]]
    elif model_type == ModelType.BERT:
        sentence_tokens = [tokenizer.tokenize(t) for t in sentence]
    else:
        raise Exception(f'Model type {model_type} not known.')

    sentence_ids = [tokenizer.convert_tokens_to_ids(t) for t in sentence_tokens]
    start_idx_mask = []
    all_ids = []
    for subwords in sentence_ids:
        curr_mask = [1]
        if len(subwords) > 1:
            curr_mask += [0] * (len(subwords) - 1)
        start_idx_mask.extend(curr_mask)
        all_ids.extend(subwords)
        assert len(start_idx_mask) == len(all_ids)
    special_token_mask = tokenizer.get_special_tokens_mask(all_ids)

    prefix_offset = 0
    while prefix_offset < len(special_token_mask) and special_token_mask[prefix_offset] == 1:
        prefix_offset += 1
    suffix_offset = len(special_token_mask) - len(start_idx_mask) - prefix_offset
    start_idx_mask = [0] * prefix_offset + start_idx_mask + [0] * suffix_offset

    sentence_inputs = tokenizer.prepare_for_model(all_ids, add_special_tokens=True)
    input_ids = sentence_inputs["input_ids"]
    attention_mask = sentence_inputs["attention_mask"]
    #######
    # test the tokenization
    inputs = tokenizer(
        text=' '.join(sentence),
        add_special_tokens=True
    )
    assert inputs["input_ids"] == input_ids
    assert inputs["attention_mask"] == attention_mask
    #######
    return input_ids, attention_mask, start_idx_mask

class UnpooledDataset(Dataset):
    def __getitem__(self, n):
        return self._features[n]

    def __len__(self):
        return len(self._features)

    def __init__(self, filepath, model_type, tokenizer):
        super(UnpooledDataset, self).__init__()
        self.init_dataset(filepath, model_type, tokenizer)

    def init_dataset(self, filepath, model_type, tokenizer):
        max_seq_len = -1
        features = []
        for line in open(filepath):
            if not line:
                continue
            example = json.loads(line)
            head_idx = example['heads_index']
            pp_idx = example['pp_index']
            children_idx = example['children_index']
            sentence = example['sentence']
            label = example['label']
            n_heads = len(head_idx)
            input_ids, attention_mask, start_idx_mask = tokenize_sentence(sentence, model_type, tokenizer)

            max_seq_len = max(max_seq_len, len(input_ids))
            datum = {
                'input_ids': torch.LongTensor(input_ids),
                'attention_mask': torch.LongTensor(attention_mask),
                'head_idx': torch.LongTensor(head_idx),
                'start_idx_mask': torch.BoolTensor(start_idx_mask),

                'n_heads': torch.LongTensor([n_heads]),
                'pp_idx': torch.LongTensor([pp_idx]),
                'children_idx': torch.LongTensor([children_idx]),
                'labels': torch.LongTensor([label])
            }
            features.append(datum)
        print(f'max_seq_len {max_seq_len}')
        self._features = features

def variable_collate_fn(batch):
    batch_features = {}

    batch_features['input_ids'] = pad_sequence([x['input_ids'] for x in batch],
                                               batch_first=True,
                                               padding_value=0)
    batch_features['attention_mask'] = pad_sequence([x['attention_mask'] for x in batch],
                                               batch_first=True,
                                               padding_value=0)
    batch_features['start_idx_mask'] = pad_sequence([x['start_idx_mask'] for x in batch],
                                               batch_first=True,
                                               padding_value=0)
    batch_features['head_idx'] = pad_sequence([x['head_idx'] for x in batch],
                                               batch_first=True,
                                               padding_value=0)

    batch_features['pp_idx'] = torch.cat([x['pp_idx'] for x in batch])
    batch_features['children_idx'] = torch.cat([x['children_idx'] for x in batch])
    batch_features['n_heads'] = torch.cat([x['n_heads'] for x in batch])

    if 'labels' in batch[0]:
        batch_features['labels'] = torch.cat([x['labels'] for x in batch])
    return batch_features
