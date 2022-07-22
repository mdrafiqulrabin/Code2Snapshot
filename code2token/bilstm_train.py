import os
import pathlib
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config as cf
from dataloader import get_token_vocab, CodeTokenDataset


class BiLSTMModel(nn.Module):

    def __init__(self, vocab_size, num_class):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, cf.EMBED_DIM, sparse=True)
        # token embeddings --> hidden states
        self.lstm = nn.LSTM(input_size=cf.EMBED_DIM, hidden_size=cf.HIDDEN_UNITS, dropout=cf.DROPOUT_RATE,
                            num_layers=cf.N_LAYERS, bidirectional=True, batch_first=True)
        # hidden state --> target label
        self.fc = nn.Linear(cf.N_LAYERS * cf.HIDDEN_UNITS, num_class)
        self.act = nn.LogSoftmax(dim=1)
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-cf.WEIGHT_RANGE, cf.WEIGHT_RANGE)
        self.fc.weight.data.uniform_(-cf.WEIGHT_RANGE, cf.WEIGHT_RANGE)
        self.fc.bias.data.zero_()

    def forward(self, tokens, offsets):
        embeds = self.embedding(tokens, offsets)
        # batch_first, EmbeddingBag(mode=mean), input_size
        outputs, _ = self.lstm(embeds.view(embeds.shape[0], 1, -1))
        scores = self.fc(outputs.view(embeds.shape[0], -1))  # [-1]
        return self.act(scores)  # ff.log_softmax(scores, dim=1)


def collate_batch(batch):
    path_list, label_list, embed_list, offset_list = [], [], [], [0]
    for (_id, _token, _embed, _label) in batch:
        path_list.append(_id)
        label_list.append(_label)
        processed_embed = torch.tensor(_embed, dtype=torch.int64)
        embed_list.append(processed_embed)
        offset_list.append(processed_embed.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    embed_list = torch.cat(embed_list)
    offset_list = torch.tensor(offset_list[:-1]).cumsum(dim=0)
    return path_list, label_list, embed_list, offset_list


def train(dataloader, model):
    model.train()
    total_acc, total_loss, total_count = 0, 0, 1e-9
    for idx, data in enumerate(dataloader):
        optimizer.zero_grad()
        labels_, embeds_, offsets_ = data[1].to(device), data[2].to(device), data[3].to(device)
        predicts_ = model(embeds_, offsets_)
        loss = criterion(predicts_, labels_)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicts_.argmax(1) == labels_).sum().item()
        total_loss += loss.item()
        total_count += labels_.size(0)
    return (total_acc / total_count) * 100, (total_loss / total_count) * 100


def evaluate(dataloader, model):
    model.eval()
    total_acc, total_loss, total_count = 0, 0, 1e-9
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            labels_, embeds_, offsets_ = data[1].to(device), data[2].to(device), data[3].to(device)
            predicts_ = model(embeds_, offsets_)
            loss = criterion(predicts_, labels_)
            total_acc += (predicts_.argmax(1) == labels_).sum().item()
            total_loss += loss.item()
            total_count += labels_.size(0)
    return (total_acc / total_count) * 100, (total_loss / total_count) * 100


# Args
args = sys.argv
curr_db_name, curr_tok_type = args[1], args[2]

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device = {}".format(device))

# Vocabulary
lookup_train_path = "{}/{}/{}.json".format(cf.ROOT_PATH, cf.DB_NAMES[curr_db_name], cf.PARTITIONS["train"])
lookup_val_path = "{}/{}/{}.json".format(cf.ROOT_PATH, cf.DB_NAMES[curr_db_name], cf.PARTITIONS["val"])
token_vocab = get_token_vocab([lookup_train_path, lookup_val_path], cf.TOKEN_TYPES[curr_tok_type])
print("vocab = #{}".format(len(token_vocab[0])))

# DataLoader
train_set = CodeTokenDataset(lookup_train_path, cf.TOKEN_TYPES[curr_tok_type], token_vocab, None)
top_labels = train_set.get_top_labels()  # required for label2index/index2label mapping
train_loader = DataLoader(train_set, batch_size=cf.BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
print("train_set = #{}".format(len(train_set)))
val_set = CodeTokenDataset(lookup_val_path, cf.TOKEN_TYPES[curr_tok_type], token_vocab, top_labels)
val_loader = DataLoader(val_set, batch_size=cf.BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
print("val_set = #{}".format(len(val_set)))

# Params
model_best_file = '/scratch/models/code2token/{}/{}_lstm_best.pth'.format(curr_tok_type, curr_db_name)
log_test_file = '/scratch/models/code2token/{}/{}_lstm_test.csv'.format(curr_tok_type, curr_db_name)
lstm_model = BiLSTMModel(len(token_vocab[0]), len(top_labels)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(lstm_model.parameters(), lr=cf.LEARNING_RATE)

# Training Model
print('\nStarted Training Model\n')
best_acc, best_ep, early_count = 0, 0, 0
for epoch in range(1, cf.MAX_EPOCH + 1):
    print("-" * 50)
    epoch_start_time = time.time()
    train_acc, train_loss = train(train_loader, lstm_model)
    val_acc, val_loss = evaluate(val_loader, lstm_model)
    epoch_end_time = time.time()
    print("time: {:.2f}s, epoch: {}".format(epoch_end_time - epoch_start_time, epoch))
    print("train acc: {:.2f}, train loss: {:.2f}".format(train_acc, train_loss))
    print("val acc: {:.2f}, val loss: {:.2f}".format(val_acc, val_loss))
    if val_acc > best_acc:
        best_acc, best_ep = val_acc, epoch
        print("\nBest Accuracy = {:.2f} % at Epoch-{}".format(best_acc, best_ep))
        print('Saving model after best epoch-{}'.format(best_ep))

        if os.path.exists(model_best_file):
            os.remove(model_best_file)
        else:
            pathlib.Path(os.path.dirname(model_best_file)).mkdir(parents=True, exist_ok=True)
        torch.save(lstm_model.state_dict(), model_best_file)
        early_count = 0
    else:
        early_count += 1
    if early_count == cf.EARLY_LIMIT:
        print("\nEarly stopping as no improvements for last {} epochs.".format(cf.EARLY_LIMIT))
        print("Final Best Accuracy = {:.2f} % at Epoch-{}\n".format(best_acc, best_ep))
        break
print('\nFinished Training Model\n')

# Evaluating Test
print('\nStarted Evaluating Test\n')

# Loading Best Model
lstm_model = BiLSTMModel(len(token_vocab[0]), len(top_labels)).to(device)
lstm_model.load_state_dict(torch.load(model_best_file))
lstm_model.eval()
print("Loaded best model from {}".format(model_best_file))

# Test Dataloader
lookup_test_path = "{}/{}/{}.json".format(cf.ROOT_PATH, cf.DB_NAMES[curr_db_name], cf.PARTITIONS["test"])
test_set = CodeTokenDataset(lookup_test_path, cf.TOKEN_TYPES[curr_tok_type], token_vocab, top_labels)
test_loader = DataLoader(test_set, batch_size=cf.BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
print("test_set = #{}".format(len(test_set)))

# Count predictions for each class
correct_pred = {classname: 0 for classname in top_labels}
total_pred = {classname: 0 for classname in top_labels}

with torch.no_grad():
    with open(log_test_file, 'w') as test_f:
        test_f.write("id,label,prediction\n")
        for _, data_t in enumerate(test_loader):
            ids_t = data_t[0]
            labels_t, embeds_t, offsets_t = data_t[1].to(device), data_t[2].to(device), data_t[3].to(device)
            predicts_t = lstm_model(embeds_t, offsets_t)
            _, predicts_t = torch.max(predicts_t, 1)
            for id_t, label_t, predict_t in zip(ids_t, labels_t, predicts_t):
                test_f.write("{},{},{}\n".format(id_t, top_labels[label_t], top_labels[predict_t]))
                if label_t == predict_t:
                    correct_pred[top_labels[label_t]] += 1
                total_pred[top_labels[label_t]] += 1

# Print accuracy for each class
print("\nAccuracy for classes:")
total_correct_count, total_pred_count = 0, 0
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / (total_pred[classname] + 1e-9)
    print("Accuracy for class {} is: {:.2f} %".format(classname, accuracy))
    total_correct_count += correct_count
    total_pred_count += total_pred[classname]
accuracy = 100 * float(total_correct_count) / (total_pred_count + 1e-9)
print("\nOverall Accuracy = {:.2f} %\n".format(accuracy))

print('\nFinished Evaluating Test\n')
