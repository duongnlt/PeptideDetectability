from pandas.io import parsers
import torch
import torch.nn as nn
from vocab import Vocab
from model import RNNPeptide
from dataset import get_data_loader
import argparse
from tqdm import tqdm
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from transformers import BertTokenizer
    # tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
torch.manual_seed(30)


parser = argparse.ArgumentParser()
parser.add_argument('--train_file', default='/home/duong/Workspace/DATN/_DeepMSPeptide/training.csv', type=str)
parser.add_argument('--test_file', default='/home/duong/Workspace/DATN/_DeepMSPeptide/test.csv', type=str)

parser.add_argument('--bsize', default=32, type=int)
parser.add_argument('--epochs', default=32, type=int)
parser.add_argument('--lr', default=5e-6, type=float)
parser.add_argument('--gpu', default=True, type=bool)
parser.add_argument('--name', default='', type=str)



args = parser.parse_args()


if __name__ == '__main__':

    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

    model_name = args.name
    device = torch.device('cuda' if args.gpu  else 'cpu')
    # vocab = Vocab()

    model = RNNPeptide().to(device=device)
    train_loader = get_data_loader(dir=args.train_file, tokenizer=tokenizer, bsize=args.bsize)
    test_loader = get_data_loader(dir=args.test_file, tokenizer=tokenizer, bsize=args.bsize)

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.AdamW([{'params':model.classifier.parameters()}, {'params': model.bert.parameters(), 'lr': 5e-6}], lr=0.0001, weight_decay=1e-2)
    # optimizer_cls = torch.optim.AdamW(model.classifier.parameters(), lr=0.001, weight_decay=1e-2)

    scheduler = StepLR(optimizer, step_size=2, gamma=0.1)

    train_loss_list = []
    val_loss_list = []
    val_acc_list = []


    for epoch in range(args.epochs):
        train_loss = 0.0
        val_loss = 0.0
        correct = 0
        total = 0

        # with tqdm(train_loader, unit='batch') as pbar:
        #     for data, label in pbar:
        #         pbar.set_description(f"Epoch {epoch}")
        #         print(data)
        #         data, label = data.to(device), label.to(device)
        #         output = model(data)

        #         loss = criterion(output, label)
        #         loss.backward()

        #         optimizer.step()
        #         optimizer.zero_grad()

        #         train_loss += loss.item()

        #         pbar.set_postfix(loss=loss.item())
        #         time.sleep(0.1)
        with tqdm(train_loader, unit='batch') as pbar:
          model.train()
          for data in pbar:
            pbar.set_description(f"Epoch {epoch}")
            ins, label, mask = data['input_ids'].to(device), data['targets'].to(device), data['attention_mask'].to(device)
            output = model(ins, mask)

            loss = criterion(output, label)
            loss.backward()

            # optimizer_bert.zero_grad()
      

            # optimizer_bert.step()
            optimizer.step()
            optimizer.zero_grad()


            train_loss += loss.item()

            pbar.set_postfix(loss=loss.item())
            time.sleep(0.1)

        train_loss /= len(train_loader)
        train_loss_list.append(train_loss)
        print("Train loss: {}".format(train_loss))

        best_valid_loss = float('inf')
        with torch.no_grad():
          model.eval()
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
        val_loss /= len(test_loader)

        val_acc_list.append(100*correct/total)
        print('Val_loss: {:.4f} | Val_acc: {:.4f}'.format(val_loss, 100*correct/total))
        scheduler.step()
        # #Early stopping
        # # if epoch != 0 and val_loss > val_loss_list[-1]:
        # #     trigger += 1
        # #     if trigger >= patience:
        # #         print('Early stopping')
        # #         break
        # # else:
        # #     trigger = 0
        
        val_loss_list.append(val_loss)
        # val_acc_list.append(100*correct/total)
        # print('Val_loss: {:.2f} | Val_acc: {:.2f}'.format(val_loss, 100*correct/total))

        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            torch.save(model.state_dict(), f"/content/drive/MyDrive/RNNMSPeptide_2/weights/{model_name}-train-best.pt")
        torch.save(model.state_dict(), f'/content/drive/MyDrive/RNNMSPeptide_2/weights/{model_name}-train-last.pt')
        # scheduler.step(val_loss)
    
    with open('result.txt', 'w') as f:
        f.write('epoch,train_loss,val_loss,val_acc\n')
        for i in range(args.epochs):
            res = [i+1, train_loss_list[i], val_loss_list[i], val_acc_list[i]]
            f.write(','.join(str(r) for r in res))
            f.write('\n')

