import torch
from dataset import get_data_loader
from vocab import Vocab
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer


# class RNNPeptide(nn.Module):
#     def __init__(self, vocab, hidden_size=128, output_size=256):
#         super(RNNPeptide, self).__init__()

#         self.vocab = vocab
#         self.embedding = nn.Embedding(len(self.vocab), hidden_size)
#         self.output_size = output_size
#         self.lstm = nn.LSTM(hidden_size, output_size, num_layers=1, batch_first=True, bidirectional = True)
#         self.fc = nn.Linear(2*output_size,1)
#     def forward(self, seq):
#         seq_emb = self.embedding(seq)

#         output, (h,c) = self.lstm(seq_emb)
#         # print(output.shape)
#         # out_forward = output[range(len(output)), self.vocab.max_seq_len - 1, :self.output_size]
#         # print(out_forward, h[0, :, :].shape)
#         # out_reverse = output[:, 0, self.output_size:]
#         h = h.permute(1,0,2)
#         h = h.reshape(h.shape[0], -1)
#         print(h.shape, output.shape)

#         # out_reduced = torch.cat((out_forward, out_reverse), 1)
#         # print(out_reduced)
#         out = self.fc(h)
#         # out = torch.sigmoid(out)
#         return out




# class RNNPeptide(nn.Module):
#     def __init__(self, vocab, hidden_size=128, output_size=256):
#         super(RNNPeptide, self).__init__()

#         self.vocab = vocab
#         self.embedding = nn.Embedding(len(self.vocab), hidden_size)
#         self.output_size = output_size
#         self.lstm = nn.LSTM(hidden_size, output_size, num_layers=1, batch_first=True, bidirectional = True)
#         self.fc = nn.Linear(2*output_size,1)
#     def forward(self, seq):
#         seq_emb = self.embedding(seq)

#         output, (h,c) = self.lstm(seq_emb)
#         # print(output.shape)
#         # out_forward = output[range(len(output)), self.vocab.max_seq_len - 1, :self.output_size]
#         # print(out_forward, h[0, :, :].shape)
#         # out_reverse = output[:, 0, self.output_size:]
#         h = h.permute(1,0,2)
#         h = h.reshape(h.shape[0], -1)
#         print(h.shape, output.shape)

#         # out_reduced = torch.cat((out_forward, out_reverse), 1)
#         # print(out_reduced)
#         out = self.fc(h)
#         # out = torch.sigmoid(out)
#         return out

class RNNPeptide(nn.Module):
    def __init__(self, hidden_size=128, output_size=256):
        super(RNNPeptide, self).__init__()

        # self.vocab = vocab
        self.bert = BertModel.from_pretrained("Rostlab/prot_bert")

        # _freeze_module = [self.bert.embeddings, *self.bert.encoder.layer[:5]]

        # for module in _freeze_module:
        #     for param in module.parameters():
        #         param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Conv1d(self.bert.config.hidden_size, 512, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.MaxPool1d(3),
  
            nn.Conv1d(512, 128, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.MaxPool1d(3),

            nn.Flatten(),
            nn.Linear(2*64*8, 100),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(100,2)
        )
        # self.dropout = nn.Dropout(0.4)
        # self.output_size = self.bert.config.hidden_size
        # # self.lstm = nn.LSTM(self.bert.config.hidden_size, self.bert.config.hidden_size, num_layers=1, batch_first=True, bidirectional = True)
        # # self.fc1 = nn.Linear(self.output_size, 512)
        # # # self.fc2 = nn.Linear(512, 256)
        # # self.fc3 = nn.Linear(512, 2)
        # self.conv1 = nn.Conv1d(self.bert.config.hidden_size, 512, kernel_size=3)
        # self.max_pool1 = nn.MaxPool1d(3)
        # # self.conv2 = nn.Conv1D(512, 128, kernel_size=3)
        # self.conv2 = nn.Conv1d(512, 128, kernel_size=3)
        # self.max_pool2 = nn.MaxPool1d(3)

        # self.classifier = nn.Sequential(nn.Dropout(p=0.2),
        #                         nn.Linear(2*64*8, 2))
    def forward(self, input_ids, attention_mask):
        seq_emb = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask           
        ).last_hidden_state

        # seq_emb = self.dropout(seq_emb.last_hidden_state)
        seq_emb = seq_emb.permute(0, 2, 1)
        # out = self.max_pool1(F.relu(self.conv1(seq_emb)))
        # out = self.max_pool2(F.relu(self.conv2(out)))

        # # h = h.permute(1,0,2)
        # # h = h.reshape(h.shape[0], -1)
        # out = out.view(out.shape[0], -1)

        out = self.classifier(seq_emb)

        # out = torch.sigmoid(out)
        return out



