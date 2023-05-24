import numpy as np
import torch
from torch.autograd import Variable
from metrics import check_accuracy
import sys
from tqdm import tqdm as tqdm

from torchmetrics.classification import BinaryAccuracy

DS = False
'''
def train(model, train_loader, val_loader, n_epochs, criterion, optimizer, scheduler):
    scaler = torch.cuda.amp.GradScaler()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t = np.linspace(1, n_epochs, n_epochs)
    
    epoch_train_loss = 0 * t
    epoch_train_accuracy = 0 * t

    epoch_val_loss = 0 * t
    epoch_val_accuracy = 0 * t
    
    i,batch = next(enumerate(train_loader))
    
    loop = tqdm(range(n_epochs), 
                        total=n_epochs,
                        leave=False) 
    
    for epoch_index in range(n_epochs):
        
        train_loss = 0.0
        train_accuracy = 0.0
        model.train()
        # loop = tqdm(enumerate(train_loader), 
        #                     total=len(train_loader),
        #                     leave=True)
        metric = BinaryAccuracy().to(device=device)

        # for i,batch in loop:
            
        I1 = Variable(batch['I1'].float().to(device=device))
        I2 = Variable(batch['I2'].float().to(device=device))
        
        label = Variable(batch['label'].to(device=device))
        label = torch.squeeze(label,dim=1)
    
        optimizer.zero_grad()
        
        output = model(I1, I2)  
        
        if DS:
            label_list = batch['list']
            loss = criterion(output, label_list)
        else:
            loss = criterion(output, label.long())

        # loss.backward()
        # optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
            
        
        with torch.no_grad():
            lss = loss.item()
            train_loss += lss
            
            if DS:
                _, predicted = torch.max(output[0].data, 1)
            else:
                _, predicted = torch.max(output.data, 1)
            
            pr = (predicted.int() > 0).to(device=device)
            gt = (label.data.int() > 0).to(device=device)
            
            acc = metric(torch.flatten(pr), torch.flatten(gt))
            train_accuracy += acc
        
        #update progress bar
        loop.set_description(f'Epoch [{epoch_index+1}/{n_epochs}]')
        loop.set_postfix(train_loss = loss.item(), train_accuracy = acc.item())
        
        scheduler.step()
        continue
        
        valid_loss = 0.0
        valid_accuracy = 0.0
        model.eval()
        loop = tqdm(enumerate(val_loader), 
                    total=len(val_loader),
                    leave=True)
        
        for i,batch in loop:
            with torch.no_grad():
                I1 = Variable(batch['I1'].float().to(device=device))
                I2 = Variable(batch['I2'].float().to(device=device))
                
                label = Variable(batch['label'].to(device=device))
                label = torch.squeeze(label,dim=1)

                output = model(I1, I2)
                
                if DS:
                    label_list = batch['list'].to(device=device)
                    loss = criterion(output, label_list)
                else:
                    loss = criterion(output, label.long())
                
                _, predicted = torch.max(output.data, 1)
                
                pr = (predicted.int() > 0)
                gt = (label.data.int() > 0)
                
                acc = metric(torch.flatten(pr), torch.flatten(gt))

                valid_loss += loss.item()
                valid_accuracy += acc
            
            #update progress bar
            loop.set_description(f'Epoch [{epoch_index+1}/{n_epochs}]')
            loop.set_postfix(valid_loss = loss.item(), valid_accuracy = acc.item())

        train_loss = train_loss / len(train_loader)
        valid_loss = valid_loss / len(val_loader)
        train_accuracy = train_accuracy / len(train_loader)
        valid_accuracy = valid_accuracy / len(val_loader)

        epoch_train_loss[epoch_index] = train_loss
        epoch_val_loss[epoch_index] = valid_loss
        epoch_train_accuracy[epoch_index] = train_accuracy
        epoch_val_accuracy[epoch_index] = valid_accuracy

        scheduler.step()
        
    history = {'train_loss': epoch_train_loss, 
               'train_accuracy': epoch_train_accuracy, 
               'val_loss': epoch_val_loss, 
               'val_accuracy': epoch_val_accuracy}    
            
    return history
'''
class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True
            
def train(model, train_loader, val_loader, n_epochs, criterion, optimizer, scheduler):
    # scaler = torch.cuda.amp.GradScaler()

    # early_stopping = EarlyStopping(tolerance=3, min_delta=0.2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t = np.linspace(1, n_epochs, n_epochs)
    
    epoch_train_loss = 0 * t
    epoch_train_accuracy = 0 * t

    epoch_val_loss = 0 * t
    epoch_val_accuracy = 0 * t
    
    for epoch_index in range(n_epochs):
        
        train_loss = 0.0
        train_accuracy = 0.0
        model.train()
        loop = tqdm(enumerate(train_loader), 
                            total=len(train_loader),
                            leave=True)
        metric = BinaryAccuracy().to(device=device)

        for i,batch in loop:
            
            I1 = Variable(batch['I1'].float().to(device=device))
            I2 = Variable(batch['I2'].float().to(device=device))
            
            label = Variable(batch['label'].to(device=device))
            label = torch.squeeze(label,dim=1)
        
            optimizer.zero_grad()

            output = model(I1, I2)  
            
            if DS:
                label_list = batch['list']
                loss = criterion(output, label_list)
            else:
                loss = criterion(output, label.long())

            loss.backward()
            optimizer.step()
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
        
            with torch.no_grad():
                lss = loss.item()
                train_loss += lss
                
                # _, predicted = torch.max(output.data, 1)
                if DS:
                    _, predicted = torch.max(output[0].data, 1)
                else:
                    _, predicted = torch.max(output.data, 1)
                
                pr = (predicted.int() > 0).to(device=device)
                gt = (label.data.int() > 0).to(device=device)
                
                acc = metric(torch.flatten(pr), torch.flatten(gt))
                train_accuracy += acc
            
            #update progress bar
            loop.set_description(f'Epoch [{epoch_index+1}/{n_epochs}]')
            loop.set_postfix(train_loss = loss.item(), train_accuracy = acc.item())
        
        valid_loss = 0.0
        valid_accuracy = 0.0
        model.eval()
        loop = tqdm(enumerate(val_loader), 
                    total=len(val_loader),
                    leave=True)
        
        for i,batch in loop:
            with torch.no_grad():
                I1 = Variable(batch['I1'].float().to(device=device))
                I2 = Variable(batch['I2'].float().to(device=device))
                
                label = Variable(batch['label'].to(device=device))
                label = torch.squeeze(label,dim=1)

                output = model(I1, I2)  
 
                if DS:
                    label_list = batch['list']
                    loss = criterion(output, label_list)
                else:
                    loss = criterion(output, label.long())
                
                # _, predicted = torch.max(output.data, 1)
                if DS:
                    _, predicted = torch.max(output[0].data, 1)
                else:
                    _, predicted = torch.max(output.data, 1)
                
                pr = (predicted.int() > 0)
                gt = (label.data.int() > 0)
                
                acc = metric(torch.flatten(pr), torch.flatten(gt))

                valid_loss += loss.item()
                valid_accuracy += acc
            
            #update progress bar
            loop.set_description(f'Epoch [{epoch_index+1}/{n_epochs}]')
            loop.set_postfix(valid_loss = loss.item(), valid_accuracy = acc.item())

        train_loss = train_loss / len(train_loader)
        valid_loss = valid_loss / len(val_loader)
        train_accuracy = train_accuracy / len(train_loader)
        valid_accuracy = valid_accuracy / len(val_loader)

        epoch_train_loss[epoch_index] = train_loss
        epoch_val_loss[epoch_index] = valid_loss
        epoch_train_accuracy[epoch_index] = train_accuracy
        epoch_val_accuracy[epoch_index] = valid_accuracy
        
        # early stopping
        # early_stopping(train_loss, valid_loss)
        # if early_stopping.early_stop:
        #     print("We are at epoch:", epoch_index)
        #     break

        scheduler.step()
        
    history = {'train_loss': epoch_train_loss, 
               'train_accuracy': epoch_train_accuracy, 
               'val_loss': epoch_val_loss, 
               'val_accuracy': epoch_val_accuracy}    
            
    return history