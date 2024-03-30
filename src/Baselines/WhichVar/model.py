import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torchtext.vocab import GloVe
import json
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import re
import random

def check_and_split_camel_case(s):
    if re.match(r'^[a-z]+([A-Z][a-z]*)*$', s):
        words = re.findall('[a-z]+|[A-Z][a-z]*', s)
        return "yes", words
    else:
        return "no", s


def setup_seed(seed):
    if seed == -1:
        seed = random.randint(0, 1000)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


class Model(nn.Module):
    def __init__(self, weight):
        super(Model, self).__init__()
        vocab_size = weight.shape[0]
        self.word_embed = nn.Embedding(num_embeddings=vocab_size+1, embedding_dim=weight.shape[-1])
        self.word_embed.weight.data[:vocab_size] = weight
        self.word_embed.weight.data[vocab_size] = torch.zeros(weight.shape[-1])
        self.word_embed.weight.requires_grad = False
        
        self.rnn = nn.LSTM(100, 128, num_layers=2, bidirectional=True, batch_first=True)
        self.num_heads = 4
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=self.num_heads, batch_first=True)
        
        self.cls_layer = nn.Linear(256, 1, bias=False)
        
        
    def forward(self, sentences, lens):
        
        embeds = self.word_embed(sentences)
        outputs, _ = self.rnn(embeds)
        attn_mask=torch.zeros((sentences.size(0) * 4, sentences.size(1), sentences.size(1)), device=sentences.device).bool()
        for i, l in enumerate(lens):
            for j in range(1, self.num_heads+1):
                attn_mask[i*j][:l][:l] = True
            
        attention_embeds, _ = self.attention(outputs, outputs, outputs, attn_mask=None)
        logits = self.cls_layer(attention_embeds).squeeze(dim=-1)
        
        return logits
    
class SensDataSet(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
 
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        tuple_ = (self.data[idx], self.label[idx])
        return tuple_
    
    
def collate_fn(data_tuple):
    # data_tuple.sort(key=lambda x: len(x[0]), reverse=True)
    data = [torch.LongTensor(sq[0]) for sq in data_tuple]
    label = [torch.Tensor(sq[1]) for sq in data_tuple]
    data_length = [len(sq) for sq in data]
    data = pad_sequence(data, batch_first=True)
    label = pad_sequence(label, batch_first=True)
    return data, label, data_length    


# def evaluate(model, test_dataloader, device):
#     acc = 0
#     n = 0
#     model.eval()
#     total_pred = []
#     total_label =[]
#     for batch_x, batch_y, batch_x_len in test_dataloader:
#         batch_x = batch_x.to(device)
#         batch_y = batch_y.to(device)
#         out = model(batch_x, batch_x_len)
#         predicts = (out > 0) + 0
#         for predict, label, length in zip(predicts, batch_y, batch_x_len):
#             total_pred.append(predict[:length])
#             total_label.append(label[:length])
#     total_pred = torch.cat(total_pred).cpu().numpy()        
#     total_label = torch.cat(total_label).cpu().numpy()      
    
#     precision = precision_score(total_label, total_pred)
#     recall = recall_score(total_label, total_pred)
#     f1 = f1_score(total_label, total_pred)     
#     return {"precision" : precision, "recall" : recall, "f1" : f1}


def evaluate(model, test_dataloader, device):
    model.eval()
    precision_list = []
    recall_list = []
    f1_list = []
    predicts_list = []
    for batch_x, batch_y, batch_x_len in test_dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        out = model(batch_x, batch_x_len)
        predicts = (out > 0) + 0
        batch_x = batch_x.cpu().numpy()
        batch_y = batch_y.cpu().numpy()
        predicts = predicts.cpu().numpy()
        for x, predict, label, length in zip(batch_x, predicts, batch_y, batch_x_len):
            # print(len(x), len(predict), len(label), length)
            x, predict, label = x[:length], predict[:length], label[:length]
            
            pred_1_set = set(x[predict == 1])
            pred_0_set = set(x) - pred_1_set
            label_1_set = set(x[label == 1])
            label_0_set = set(x) - label_1_set
            TP = len(label_1_set.intersection(pred_1_set))
            FN = len(label_1_set.intersection(pred_0_set))
            FP = len(label_0_set.intersection(pred_1_set))
            TN = len(label_0_set.intersection(pred_0_set))
            precision = TP / (TP + FP) if (TP + FP) != 0 else 0
            recall = TP / (TP + FN) if (TP + FN) != 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            predicts_list.append(predict)
    
    precision = np.mean(precision_list)
    recall = np.mean(recall_list)
    f1 = np.mean(f1_list)
    # print(len(f1_list))
    
    return {"precision" : precision, "recall" : recall, "f1" : f1}, predicts_list




if __name__ == '__main__':
    setup_seed(111)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)
    epochs = 0
    glove = GloVe(name='6B', dim=100)
    vocab_size = len(glove)
    with open('./train.json', 'r') as file:
        train_data = json.load(file)
    with open('./test.json', 'r') as file:
        test_data = json.load(file)

    train_sentences = train_data['input']
    train_sentences = [[glove.stoi[word] if word in glove.stoi.keys() else vocab_size for word in sentence] for sentence in train_sentences]
    train_labels = train_data['label']
    
    test_sentences = test_data['input']
    test_sentences = [[glove.stoi[word] if word in glove.stoi.keys() else vocab_size for word in sentence] for sentence in test_sentences]
    test_labels = test_data['label']
    # print(len(test_sentences))
    # train_size = int(0.8 * len(sentences))
    # train_sentences, test_sentences = sentences[:train_size], sentences[train_size:]
    # train_labels, test_labels = labels[:train_size], labels[train_size:]
    
    train_dataset = SensDataSet(data=train_sentences, label=train_labels)
    test_dataset = SensDataSet(data=test_sentences, label=test_labels)

    # train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    model = Model(weight=glove.vectors).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-03)
    loss_fun = nn.BCEWithLogitsLoss(reduction='none')
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = []
        for batch_id, (batch_x, batch_y, batch_x_len) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            out = model(batch_x, batch_x_len)
            pos_mask=torch.zeros((batch_x.size(0), batch_x.size(1)), device=device).bool()
            for i, l in enumerate(batch_x_len):
                pos_mask[i][:l] = True
            loss = loss_fun(out, batch_y)[pos_mask].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
        print("epoch: {}/{}, loss={}".format(epoch, epochs, np.mean(total_loss)))
        result1, predicts_list1 = evaluate(model, train_loader, device)
        result2, predicts_list2 = evaluate(model, test_loader, device)
        print('result on train set: {}'.format(result1))
        print('result on test set: {}'.format(result2))

    torch.save(model.state_dict(), 'model/model.pth')
    
    model.load_state_dict(torch.load('model/model.pth'))
    
    result, predicts_list = evaluate(model, test_loader, device)
    print(len(predicts_list))
    test_cases = []
    for i in range(len(predicts_list)):
        codes = test_data['codes'][i]
        predict = predicts_list[i]
        # print(len(test_data['input'][i]), len(test_sentences[i]), len(predict))
        variables = list(set([test_data['input'][i][j] for j, v in enumerate(predict) if v == 1]))
        label_variables = test_data['variables'][i]
        output_data = {
            'code': codes,
            'pred_variables': variables,
            'label_variables': label_variables
        }
        test_cases.append(output_data)
    json.dump(test_cases, open('output.json', 'w'), indent=4)
    
    