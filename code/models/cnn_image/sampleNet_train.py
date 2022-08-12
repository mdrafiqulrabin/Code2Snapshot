import os
import pathlib
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as ff
import torch.optim as optim
from torch.utils.data import DataLoader

import config as cf
from dataloader import CodeDataset

torch.cuda.empty_cache()

# ------------- Params -------------
args = sys.argv
curr_db_name, curr_img_type = args[1], args[2]
model_best_file = '/scratch/models/code2img/{}/{}_sampleNet_best.pth'.format(curr_img_type, curr_db_name)
log_test_file = '/scratch/models/code2img/{}/{}_sampleNet_test.csv'.format(curr_img_type, curr_db_name)

# ------------- DataLoader -------------
print("\nDataLoader:\n")
train_set = CodeDataset(cf.ROOT_PATH, cf.IMG_TYPES[curr_img_type], cf.DB_NAMES[curr_db_name],
                        cf.PARTITIONS["train"], None)
top_labels = train_set.get_top_labels()  # required for label2index/index2label mapping
train_loader = DataLoader(train_set, batch_size=cf.BATCH_SIZE, shuffle=True, num_workers=cf.NUM_WORKER)
print("train_set = #{}".format(len(train_set)))
val_set = CodeDataset(cf.ROOT_PATH, cf.IMG_TYPES[curr_img_type], cf.DB_NAMES[curr_db_name],
                      cf.PARTITIONS["val"], top_labels)
val_loader = DataLoader(val_set, batch_size=cf.BATCH_SIZE, shuffle=False, num_workers=cf.NUM_WORKER)
print("val_set = #{}".format(len(val_set)))

# ------------- Config -------------
print("\nConfiguration:\n")
print("DB_NAME = {}, IMG_TYPE = {}, TRANS_SIZE = {}".format(curr_db_name, curr_img_type, cf.IMG_TRANS_SIZE))
print("MAX_EPOCH = {}, BATCH_SIZE = {}".format(cf.MAX_EPOCH, cf.BATCH_SIZE))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("\ndevice = {}\n".format(device))


# ------------- Sample Net -------------
# todo: replace with AlexNet/RestNet model class (refs.txt)
class SampleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(cf.SIZE_PARAMS[cf.IMG_TRANS_SIZE], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, len(top_labels))

    def forward(self, x):
        x = self.pool(ff.relu(self.conv1(x)))
        x = self.pool(ff.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = ff.relu(self.fc1(x))
        x = ff.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ------------- Initialize -------------
sampleNet_model = SampleNet()
sampleNet_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(sampleNet_model.parameters(), lr=0.001, momentum=0.9)

print('\nStarted Training Model\n')
best_ep, best_acc, early_count = 0, 0, 0
for epoch in range(1, cf.MAX_EPOCH + 1):
    print("-" * 50)
    epoch_start_time = time.time()

    # ------------- Training -------------
    sampleNet_model.train()
    total_train_correct, total_train_loss = 0, 0
    for i, data in enumerate(train_loader, 0):
        # data as [ids, inputs, labels]
        inputs, labels = data[1].to(device), data[2].to(device)
        # zero gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = sampleNet_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # calculate loss
        running_loss = loss.item()
        total_train_loss += running_loss
        # count correct predictions
        _, predictions = torch.max(outputs, 1)
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                total_train_correct += 1
    train_acc, train_loss = total_train_correct / len(train_loader), total_train_loss / len(train_loader)

    # ------------- Evaluating -------------
    sampleNet_model.eval()
    # no_grad when not training
    total_val_correct, total_val_loss = 0, 0
    with torch.no_grad():
        for data in val_loader:
            # x/y to device
            images, labels = data[1].to(device), data[2].to(device)
            # calculate outputs
            outputs = sampleNet_model(images)
            # calculate loss
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
            # count correct predictions
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    total_val_correct += 1
    val_acc, val_loss = total_val_correct / len(val_loader), total_val_loss / len(val_loader)

    epoch_end_time = time.time()

    print("time: {:.2f}s, epoch: {}".format(epoch_end_time - epoch_start_time, epoch))
    print("train acc: {:.2f}, train loss: {:.2f}".format(train_acc, train_loss))
    print("val acc: {:.2f}, val loss: {:.2f}".format(val_acc, val_loss))

    # check for best epoch
    if val_acc > best_acc:
        best_acc, best_ep = val_acc, epoch
        print("\nBest Accuracy = {:.2f} % at Epoch-{}".format(best_acc, best_ep))
        print('Saving model after best epoch-{}'.format(best_ep))

        if os.path.exists(model_best_file):
            os.remove(model_best_file)
        else:
            pathlib.Path(os.path.dirname(model_best_file)).mkdir(parents=True, exist_ok=True)
        torch.save(sampleNet_model.state_dict(), model_best_file)
        early_count = 0
    else:
        early_count += 1
    if early_count == cf.EARLY_LIMIT:
        print("\nEarly stopping as no improvements for last {} epochs.".format(cf.EARLY_LIMIT))
        print("Final Best Accuracy = {:.2f} % at Epoch-{}\n".format(best_acc, best_ep))
        break
print('\nFinished Training Model\n')

# ------------- Testing -------------
print('\nStarted Evaluating Test\n')

# Loading Best Model
sampleNet_model = SampleNet()
sampleNet_model.to(device)
sampleNet_model.load_state_dict(torch.load(model_best_file))
sampleNet_model.eval()
print("Loaded best model from {}".format(model_best_file))

# Test Dataloader
test_set = CodeDataset(cf.ROOT_PATH, cf.IMG_TYPES[curr_img_type], cf.DB_NAMES[curr_db_name],
                       cf.PARTITIONS["test"], top_labels)
test_loader = DataLoader(test_set, batch_size=cf.BATCH_SIZE, shuffle=False, num_workers=cf.NUM_WORKER)
print("test_set = #{}".format(len(test_set)))

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in top_labels}
total_pred = {classname: 0 for classname in top_labels}

# no_grad as not training
with torch.no_grad():
    with open(log_test_file, 'w') as test_f:
        test_f.write("id,label,prediction\n")
        for data in test_loader:
            ids, images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = sampleNet_model(images)
            _, predictions = torch.max(outputs, 1)
            for id_, label, prediction in zip(ids, labels, predictions):
                test_f.write("{},{},{}\n".format(id_, top_labels[label], top_labels[prediction]))
                if label == prediction:
                    correct_pred[top_labels[label]] += 1
                total_pred[top_labels[label]] += 1

# print accuracy for each class
print("\nAccuracy for classes:")
total_correct_count, total_pred_count = 0, 0
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / (total_pred[classname] + 1e-9)
    print("Accuracy for class {} is: {:.2f} %".format(classname, accuracy))
    total_correct_count += correct_count
    total_pred_count += total_pred[classname]
accuracy = 100 * float(total_correct_count) / (total_pred_count + 1e-9)
print("\nOverall Accuracy = {:.2f} %\n".format(accuracy))

print('Finished Evaluating Test\n')
