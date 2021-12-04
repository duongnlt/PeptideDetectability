import torch
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import pandas as pd

# class PeptideData(Dataset):
#     def __init__(self, dir, vocab):
#         super(PeptideData, self).__init__()

#         self.dir = dir
#         self.vocab = vocab
#         df_seq = pd.read_csv(self.dir)
#         self.data, self.label = df_seq['Peptide_seq'].values, df_seq['Class'].values
#         # self.label[self.label == 'MObs'] = 0
#         # self.label[self.label == 'LObs'] = 1
        
    
#     def __len__(self):
#         return len(self.data)
    

#     def __getitem__(self, idx):
#         return torch.as_tensor(self.vocab.encode(self.data[idx]), dtype=torch.long), torch.as_tensor(self.label[idx], dtype=torch.long)
    
# def get_data_loader(dir, vocab,bsize=32, n_workers=4):
#     return DataLoader(dataset=PeptideData(dir=dir, vocab=vocab), batch_size=bsize, shuffle=True, num_workers=n_workers)

# if __name__ == '__main__':
#     data_loader = get_data_loader(dir='../training.csv', vocab=Vocab(), bsize=32)
#     for batch, (data, label) in enumerate(data_loader):
#         print(data.shape)
class PeptideData(Dataset):
    def __init__(self, dir, tokenizer):
        super(PeptideData, self).__init__()

        self.dir = dir
        # self.vocab = vocab
        df_seq = pd.read_csv(self.dir)
        self.data, self.label = df_seq['Peptide_seq'].values, df_seq['Class'].values
        # self.label[self.label == 'MObs'] = 0
        # self.label[self.label == 'LObs'] = 1
        self.tokenizer = tokenizer
        self.max_len = 81
        
    
    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        sequence = str(self.data[idx])
        sequence = " ".join(sequence)

        target = self.label[idx]
        encoding = self.tokenizer.encode_plus(
            sequence,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
          'protein_sequence': sequence,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'targets': torch.tensor(target, dtype=torch.long)
        }

def my_collate(batch):
    sequences = [item[0] for item in batch]
    input_ids = [item[1] for item in batch]
    attention_masks = [item[2] for item in batch]
    targets = [item[3] for item in batch]
    return sequences, torch.Tensor(sequences), torch.Tensor(attention_masks), torch.Tensor(targets)
    
def get_data_loader(dir, tokenizer ,bsize=32, n_workers=4):
    return DataLoader(dataset=PeptideData(dir=dir, tokenizer=tokenizer), batch_size=bsize, shuffle=True, num_workers=n_workers)

if __name__ == '__main__':
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    data_loader = get_data_loader(dir='../../training.csv', tokenizer=tokenizer, bsize=32)
    for batch, data in enumerate(data_loader):
        print(data.keys())
        import time
        time.sleep(100)