import os,copy,argparse
import numpy as np
import pandas as pd
import torch,torch.nn as nn,torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score,accuracy_score
from src.model import NeuroCNN,NeuroResNet,NeuroTCN
from src.preprocessing import SignalScaler

BATCH_SIZE=64
LEARNING_RATE=0.001
PATIENCE=12
EPOCHS=60
ARTIFACTS_DIR="artifacts"

class EMGDataset(Dataset):
    def __init__(self,X,y):
        self.X=torch.FloatTensor(X)
        self.y=torch.LongTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self,idx): return self.X[idx],self.y[idx]

def augment_data(x):
    device=x.device
    scale=1.0+torch.randn(x.shape[0],1,x.shape[2],device=device)*0.1
    noise=torch.randn_like(x)*0.02
    return (x*scale)+noise

def train_one_epoch(model,loader,criterion,optimizer,device):
    model.train()
    running_loss=0.0
    for X_batch,y_batch in loader:
        X_batch,y_batch=X_batch.to(device),y_batch.to(device)
        X_aug=augment_data(X_batch)
        optimizer.zero_grad()
        outputs=model(X_aug)
        loss=criterion(outputs,y_batch)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()*X_batch.size(0)
    return running_loss/len(loader.dataset)

def validate(model,loader,criterion,device):
    model.eval()
    running_loss=0.0
    preds,targets=[],[]
    with torch.no_grad():
        for X_batch,y_batch in loader:
            X_batch,y_batch=X_batch.to(device),y_batch.to(device)
            outputs=model(X_batch)
            loss=criterion(outputs,y_batch)
            running_loss+=loss.item()*X_batch.size(0)
            preds.extend(torch.argmax(outputs,dim=1).cpu().numpy())
            targets.extend(y_batch.cpu().numpy())
    f1= f1_score(targets,preds,average='macro')
    acc=accuracy_score(targets,preds)
    return running_loss/len(loader.dataset),f1,acc

def run_training(model_name,device):
    os.makedirs(ARTIFACTS_DIR,exist_ok=True)
    X=np.load('data/processed/X_all.npy')
    y=np.load('data/processed/y_all.npy')
    groups=np.load('data/processed/groups_all.npy')
    gkf=GroupKFold(n_splits=5)
    history=[]
    for fold,(train_idx,val_idx) in enumerate(gkf.split(X,y,groups)):
        X_train,X_val=X[train_idx],X[val_idx]
        y_train,y_val=y[train_idx],y[val_idx]
        scaler=SignalScaler()
        scaler.fit(X_train)
        X_train,X_val=scaler.transform(X_train),scaler.transform(X_val)
        scaler.save(os.path.join(ARTIFACTS_DIR,f'scaler_{model_name}_fold_{fold}.json'))
        train_loader=DataLoader(EMGDataset(X_train,y_train),batch_size=BATCH_SIZE,shuffle=True,num_workers=2)
        val_loader=DataLoader(EMGDataset(X_val,y_val),batch_size=BATCH_SIZE,shuffle=False,num_workers=2)
        if model_name=='cnn': model=NeuroCNN()
        elif model_name=='resnet': model=NeuroResNet()
        elif model_name=='tcn': model=NeuroTCN()
        model=model.to(device)
        optimizer=optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=1e-3)
        criterion=nn.CrossEntropyLoss(label_smoothing=0.1)
        scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=0.5,patience=3)
        best_f1=0.0
        patience_counter=0
        best_weights=copy.deepcopy(model.state_dict())
        for epoch in range(EPOCHS):
            train_loss=train_one_epoch(model,train_loader,criterion,optimizer,device)
            val_loss,val_f1,val_acc=validate(model,val_loader,criterion,device)
            scheduler.step(val_f1)
            if val_f1>best_f1:
                best_f1=val_f1
                best_weights=copy.deepcopy(model.state_dict())
                patience_counter=0
            else:
                patience_counter+=1
                if patience_counter%5==0:
                    print(f"Ep {epoch+1:02d} Patience {patience_counter}/{PATIENCE} F1: {val_f1:.4f}")
            if patience_counter>=PATIENCE:
                break
        torch.save(best_weights,os.path.join(ARTIFACTS_DIR,f'model_{model_name}_fold_{fold}.pth'))
        history.append({'fold':fold,'best_f1':best_f1,'val_acc':val_acc})
    pd.DataFrame(history).to_csv(os.path.join(ARTIFACTS_DIR,f'training_log_{model_name}.csv'),index=False)

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=str,required=True,choices=['cnn','resnet','tcn'])
    args=parser.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_training(args.model,device)

