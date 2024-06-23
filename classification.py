import torch
import json
import csv
from tqdm import tqdm
from pprint import pprint
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import os
from torch import nn, optim
import regex as re
print("Done importing...")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = text.strip()
    text = re.sub('\s+', ' ', text)
    return text

class lstm(nn.Module):
    def __init__(self, vocab_size):
        super(lstm, self).__init__()
        self.vocab_size = vocab_size
        # lstm with 2 stacks
        num_stacks = 2
        self.embeddings = nn.Embedding(vocab_size, 300)
        self.lstm = nn.LSTM(300, 300, 1, batch_first=True)
        self.lstm1 = nn.LSTM(300, 300, 1, batch_first=True)
        self.linear = nn.Linear(300, vocab_size)
        
    def forward(self, x):
        embed = self.embeddings(x)
        x1, _ = self.lstm(embed)
        x2, _ = self.lstm1(x1)
        x = self.linear(x2)
        return x, (embed, x1, x2)

def make_dataset(data, word2idx, idx2word, max_sentences=20000):
    seq_list = []
    label_list = []
    length_list = []
    data = data[1:max_sentences+1] 
    for item in data:
        idx_seq = []
        item = item.strip().split(',') 
        label = item[0]
        seq = preprocess(item[1])
        seq = seq.split()
        for word in seq:
            if word in word2idx.keys():
                idx_seq.append(word2idx[word])
            else:
                idx_seq.append(word2idx["unk"])
        length = len(idx_seq)
        length_list.append(length)
        seq_list.append(idx_seq)
        label_list.append(label)
    return seq_list, label_list, length_list


import os
import torch

word2idx_path = "word2idx.pt"
idx2word_path = "idx2word.pt"

word2idx = torch.load(word2idx_path)
idx2word = torch.load(idx2word_path)

def load_dataset(filename):
    file_path = os.path.join("data", filename)
    with open(file_path, "r") as file:
        data = file.readlines()
    return data

train_data = load_dataset("train.csv")
test_data = load_dataset("test.csv")

# Preprocess and create sequences, labels, and lengths
train_seqs, train_labels, train_lengths = make_dataset(train_data, word2idx, idx2word)
test_seqs, test_labels, test_lengths = make_dataset(test_data, word2idx, idx2word)

from torch.nn.utils.rnn import pad_sequence

train_seqs = pad_sequence([torch.LongTensor(i) for i in train_seqs], batch_first=True)
test_seqs = pad_sequence([torch.LongTensor(i) for i in test_seqs], batch_first=True)

train_labels = torch.LongTensor([int(i) for i in train_labels])
test_labels = torch.LongTensor([int(i) for i in test_labels])
train_lengths = torch.LongTensor(train_lengths)
test_lengths = torch.LongTensor(test_lengths)

print("train, test input shapes: ", train_seqs.shape, test_seqs.shape)
print("train, test label shapes: ", train_labels.shape, test_labels.shape)
print("train, test length shapes: ", train_lengths.shape, test_lengths.shape)

class Classifier(nn.Module):
    def __init__(self, num_classes, in_dim):
        super(Classifier, self).__init__()
        self.linear_layer = nn.Linear(in_dim, 600)
        self.lstm = nn.LSTM(600, 300, 1, batch_first=True)
        self.linear_layer2 = nn.Linear(300, num_classes)
        
        # Initialize learnable lambda parameters
        self.lambda_0 = nn.Parameter(torch.rand(1))
        self.lambda_1 = nn.Parameter(torch.rand(1))
        self.lambda_2 = nn.Parameter(torch.rand(1))

    def forward(self, x0, x1, x2, length):
        x = self.linear_layer(self.lambda_0 * x0 + self.lambda_1 * x1 + self.lambda_2 * x2)
        x, _ = self.lstm(x)
       
        final_preds = []
        loc_cnt = 0
        for loc_len in length:
            loc_x = x[loc_cnt, loc_len-1, :]
            loc_out = self.linear_layer2(loc_x)
            loc_cnt += 1
            final_preds.append(loc_out)
        final_preds = torch.stack(final_preds)
        return final_preds
    
forward_lstm = torch.load("backward_model_final.pt").to(device)
backward_lstm = torch.load("forward_model_final.pt").to(device)

#freeze the ELMo model parameters
for param in list(forward_lstm.parameters()) + list(backward_lstm.parameters()):
    param.requires_grad = False

train_dataset = torch.utils.data.TensorDataset(train_seqs, train_labels, train_lengths)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

test_dataset = torch.utils.data.TensorDataset(test_seqs, test_labels, test_lengths)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=50, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=50, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=50, shuffle=True)

in_num = 600
num_classes = 5
downstream_model = Classifier(num_classes, in_num).to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(downstream_model.parameters(), lr=0.0001)

from tqdm import tqdm

n_epochs = 15
best_val_f1 = 0.0

for epoch in range(n_epochs):
    train_accurs = []
    train_f1s = []
    val_accurs = []
    val_f1s = []
    
    print("Epoch {}:".format(epoch + 1))
    
    # Training
    downstream_model.train()
    train_loader_tqdm = tqdm(train_loader, desc="Training")
    for data, label, length in train_loader_tqdm:
        data = data.to(device)
        data_reverse = data.flip(1)
        _, (xf0, xf1, xf2) = forward_lstm(data)
        _, (xb0, xb1, xb2) = backward_lstm(data_reverse)
        xf0, xf1, xf2 = xf0.detach(), xf1.detach(), xf2.detach()
        xb0, xb1, xb2 = xb0.detach(), xb1.detach(), xb2.detach()
        x0 = torch.cat((xf0, xb0), dim=2)
        x1 = torch.cat((xf1, xb1), dim=2)
        x2 = torch.cat((xf2, xb2), dim=2)
        output = downstream_model(x0, x1, x2, length)
        loss = loss_func(output, label.to(device))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(output, 1)
        accuracy = accuracy_score(label.cpu(), predicted.cpu())
        micro_f1 = f1_score(label.cpu(), predicted.cpu(), average="micro")
        train_accurs.append(accuracy)
        train_f1s.append(micro_f1)
        train_loader_tqdm.set_postfix(loss=loss.item(), accuracy=accuracy, micro_f1=micro_f1)
    
    train_accuracy = sum(train_accurs) / len(train_accurs)
    train_micro_f1 = sum(train_f1s) / len(train_f1s)
    print("Training - Accuracy: {:.4f}, Micro F1: {:.4f}".format(train_accuracy, train_micro_f1))
    
    # Validation
    downstream_model.eval()
    with torch.no_grad():
        val_loader_tqdm = tqdm(val_loader, desc="Validation")
        for data, label, length in val_loader_tqdm:
            data = data.to(device)
            data_reverse = data.flip(1)
            _, (xf0, xf1, xf2) = forward_lstm(data)
            _, (xb0, xb1, xb2) = backward_lstm(data_reverse)
            xf0, xf1, xf2 = xf0.detach(), xf1.detach(), xf2.detach()
            xb0, xb1, xb2 = xb0.detach(), xb1.detach(), xb2.detach()
            x0 = torch.cat((xf0, xb0), dim=2)
            x1 = torch.cat((xf1, xb1), dim=2)
            x2 = torch.cat((xf2, xb2), dim=2)
            output = downstream_model(x0, x1, x2, length)

            _, predicted = torch.max(output, 1)
            accuracy = accuracy_score(label.cpu(), predicted.cpu())
            micro_f1 = f1_score(label.cpu(), predicted.cpu(), average="micro")
            val_accurs.append(accuracy)
            val_f1s.append(micro_f1)
            val_loader_tqdm.set_postfix(accuracy=accuracy, micro_f1=micro_f1)
    
    val_accuracy = sum(val_accurs) / len(val_accurs)
    val_micro_f1 = sum(val_f1s) / len(val_f1s)
    print("Validation - Accuracy: {:.4f}, Micro F1: {:.4f}".format(val_accuracy, val_micro_f1))
    
    if val_micro_f1 > best_val_f1:
        best_val_f1 = val_micro_f1
        # Save the model
        torch.save(downstream_model, "classifier.pt")

print("Training complete.")

# Testing the best model
best_downstream_model = torch.load("classifier.pt")
best_downstream_model.eval()

test_accurs = []
test_f1s = []

with torch.no_grad():
    for data, label, length in test_loader:
        data = data.to(device)
        data_reverse = data.flip(1)
        _, (xf0, xf1, xf2) = forward_lstm(data)
        _, (xb0, xb1, xb2) = backward_lstm(data_reverse)
        xf0, xf1, xf2 = xf0.detach(), xf1.detach(), xf2.detach()
        xb0, xb1, xb2 = xb0.detach(), xb1.detach(), xb2.detach()
        x0 = torch.cat((xf0, xb0), dim=2)
        x1 = torch.cat((xf1, xb1), dim=2)
        x2 = torch.cat((xf2, xb2), dim=2)
        output = best_downstream_model(x0, x1, x2, length)
        
        _, predicted = torch.max(output, 1)
        accuracy = accuracy_score(label.cpu(), predicted.cpu())
        micro_f1 = f1_score(label.cpu(), predicted.cpu(), average="micro")
        test_accurs.append(accuracy)
        test_f1s.append(micro_f1)

test_accuracy = sum(test_accurs) / len(test_accurs)
test_micro_f1 = sum(test_f1s) / len(test_f1s)
print("Test - Accuracy: {:.4f}, Micro F1: {:.4f}".format(test_accuracy, test_micro_f1))