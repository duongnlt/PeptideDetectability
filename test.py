from pandas.io import parsers
import torch
import torch.nn as nn
from vocab import Vocab
from model import RNNPeptide
from dataset import get_data_loader
import argparse
from tqdm import tqdm
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import BertTokenizer
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

    # tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
torch.manual_seed(30)

parser = argparse.ArgumentParser()
# parser.add_argument('--train_file', default='/home/duong/Workspace/DATN/_DeepMSPeptide/training.csv', type=str)
parser.add_argument('--file', default='/home/duong/Workspace/DATN/_DeepMSPeptide/test.csv', type=str)

parser.add_argument('--bsize', default=32, type=int)
# parser.add_argument('--epochs', default=32, type=int)
# parser.add_argument('--lr', default=5e-6, type=float)
parser.add_argument('--gpu', default=True, type=bool)
parser.add_argument('--weight', default='', type=str, required=True)


args = parser.parse_args()


if __name__ == '__main__':

    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    test_loader = get_data_loader(dir=args.file, tokenizer=tokenizer, bsize=args.bsize)
    device = torch.device('cuda' if args.gpu  else 'cpu')

    model = RNNPeptide().to(device=device)
    model.load_state_dict(torch.load(args.weight))
    model.eval()

    criterion = nn.CrossEntropyLoss()

    preds = []
    targets = []

    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(test_loader, unit='batch') as pbar:
            for data in pbar:
                ins, label, mask = data['input_ids'].to(device), data['targets'].to(device), data['attention_mask'].to(device)
                output = model(ins, mask)

                loss = criterion(output, label)

                val_loss += loss.item()
                _, predicted = torch.max(output, 1)
                
                total += label.size(0)
                correct += (predicted == label).sum().item()
                pbar.set_postfix(loss=loss.item())
                preds.append(predicted.cpu().numpy())
                targets.append(label.cpu().numpy())
    print(preds[0].shape)
    val_loss /= len(test_loader)
    print('Val_loss: {:.4f} | Val_acc: {:.4f}'.format(val_loss, 100*correct/total))
    preds = np.hstack(preds)
    targets = np.hstack(targets)

    print(f"Precision: {precision_score(targets, preds)} | Recall: {recall_score(targets, preds)}")
    