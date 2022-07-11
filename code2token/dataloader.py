import json

import torch
from torch.utils.data import Dataset, DataLoader

import config as cf


def get_input_token(all_tokens, token_type):
    def _todo_(t):
        return t.replace(' ', '').lower()
    input_token = []
    if token_type == cf.TOKEN_TYPES["value"]:
        input_token = [_todo_(t[0]) for t in all_tokens]
    elif token_type == cf.TOKEN_TYPES["kind"]:
        input_token = [_todo_(t[1]) for t in all_tokens]
    return input_token


def get_token_vocab(lookup_paths, token_type):
    token_vocab = []
    for lookup_path in lookup_paths:
        with open(lookup_path) as f_json:
            for entry in f_json:
                json_obj = json.loads(entry.strip())
                cur_tokens = get_input_token(json_obj["tokens"], token_type)
                token_vocab.extend(cur_tokens)
    token_vocab = sorted(list(set(token_vocab)))
    token2index, index2token = {"<UNK>": 0}, {0: "<UNK>"}
    for idx, val in enumerate(token_vocab):
        token2index[val] = idx + 1
        index2token[idx + 1] = val
    return token2index, index2token


def get_xy(lookup_path, token_type):
    ids, tokens, labels = [], {}, {}
    with open(lookup_path) as f_json:
        for entry in f_json:
            json_obj = json.loads(entry.strip())
            cur_label = json_obj["method"]
            cur_path = json_obj["path"]
            cur_token = get_input_token(json_obj["tokens"], token_type)
            ids.append(cur_path)
            tokens[cur_path] = cur_token
            labels[cur_path] = cur_label
    return ids, tokens, labels


class CodeTokenDataset(Dataset):
    def __init__(self, lookup_path, token_type, token_vocab, top_labels=None):
        self.token2index, self.index2token = token_vocab
        self.ids, self.tokens, self.labels = get_xy(lookup_path, token_type)
        if not top_labels:
            self.top_labels = sorted(list(set(self.labels.values())))
        else:
            self.top_labels = sorted(top_labels)
        self.label2index = {m: i for i, m in enumerate(self.top_labels)}
        self.index2label = {i: m for i, m in enumerate(self.top_labels)}
        for k, v in self.labels.items():
            self.labels[k] = self.label2index[v]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        token_id = self.ids[index]
        input_x = self.tokens[token_id]
        input_emb = self.get_token_embed(input_x)
        output_y = self.labels[token_id]
        return token_id, input_x, input_emb, output_y

    def get_token_embed(self, token):
        return [self.token2index.get(t, self.token2index["<UNK>"]) for t in token]

    def get_top_labels(self):
        return self.top_labels


def my_collate_fn(batch):
    path_list, token_list, embed_list, label_list, offset_list = [], [], [], [], [0]
    for (_id, _token, _embed, _label) in batch:
        path_list.append(_id)
        token_list.append(_token)
        processed_embed = torch.tensor(_embed, dtype=torch.int64)
        embed_list.append(processed_embed)
        label_list.append(_label)
        offset_list.append(processed_embed.size(0))
    embed_list = torch.cat(embed_list)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offset_list = torch.tensor(offset_list[:-1]).cumsum(dim=0)
    return path_list, token_list, embed_list, label_list, offset_list


if __name__ == '__main__':
    debug_ = False
    if debug_:
        test_path = "{}/{}/{}.json".format(cf.ROOT_PATH, cf.DB_NAMES[cf.DB_NAME], cf.PARTITIONS["test"])
        test_vocab = get_token_vocab([test_path], cf.TOKEN_TYPES[cf.TOKEN_TYPE])
        test_set = CodeTokenDataset(test_path, cf.TOKEN_TYPES[cf.TOKEN_TYPE], test_vocab)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=my_collate_fn)
        print("test set = #{}".format(len(test_set)))
        print("vocab = #{}".format(len(test_vocab[0])))
        print()

        # get some random input
        dataiter = iter(test_loader)
        ids_, tokens_, embeds_, labels_, offsets_ = next(dataiter)
        print("path = ", ids_)
        print("tokens = ", tokens_)
        print("embeds = ", embeds_)
        print("label = ", labels_, labels_.numpy()[0])
        print("offsets = ", offsets_)
    else:
        print("Debug: {}".format(debug_))
