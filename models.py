import torch
from torch import nn


class Mymodel(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits
    

def test_accuracy(model, dataloader):
    
    n_corrects = 0


    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        for image_batch, label_batch in dataloader:
            
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)


            logits_batch = model(image_batch)

            predict_batch = logits_batch.argmax(dim=1)
            n_corrects += (label_batch == predict_batch).sum().item()

    accuracy = n_corrects / len(dataloader.dataset)

    return accuracy
    
def train(model, dataloador, loss_fn, optimizer):
    """1エポックの学習を行う"""

    device = next(model.parameters()).device

    model.train()
    for image_batch, label_batch in dataloador:
        
        image_batch = image_batch.to(device)
        label_batch = label_batch.to(device)

        #モデルにバッチを入れて計算 
        logits_batch = model(image_batch)

        #損失（誤差）を計算する
        loss = loss_fn(logits_batch, label_batch)

        #最適化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()

def test(model, dataloador, loss_fn):
    """1エポックのロスを計算"""
    loss_total = 0.0

    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        for image_batch, label_batch in dataloador:

            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)
            #モデルにバッチを入れて計算 
            logits_batch = model(image_batch)

            #損失（誤差）を計算する
            loss = loss_fn(logits_batch, label_batch)
            loss_total += loss.item()

    return loss_total / len(dataloador)