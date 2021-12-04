import torch

class Vocab():
    def __init__(self):
        self.pad = 0
        self.aa = {'A':1,'R':2,'N':3,'D':4,'C':5,'Q':6,'E':7,'G':8,'H':9,'I':10,'L':11,'K':12,'M':13,'F':14, 'P':15,'O':16,'S':17,'U':18,'T':19,'W':20,'Y':21,'V':22}
        self.max_seq_len = 81
    def encode(self, seq):

        encoded = [self.aa[c] for c in seq]
        # for idx in range(len(seq)):
        #     encoded += [self.aa[c] for c in seq[idx]]
        if len(encoded) < self.max_seq_len:
            encoded += [self.pad]*(self.max_seq_len-len(encoded))
        return torch.as_tensor(encoded, dtype=torch.long)
    # def decode(self, )
    def __len__(self):
        return len(self.aa) + 1

# if __name__ == '__main__':
#     vocab = Vocab()
#     print(len(vocab))
#     print(vocab.encode('HQGVMVGMGQK'))
