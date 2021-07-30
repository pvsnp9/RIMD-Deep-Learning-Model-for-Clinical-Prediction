import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_classification
from torch.nn.modules import module
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
from src.utils.class_balanceloss import CBLoss
import seaborn as sns

torch.manual_seed(1048)
torch.cuda.manual_seed(1048)
np.random.seed(1048)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class MyNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(60, 10)
        self.l2 = nn.Linear(10,2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.01)
        self.softmax = nn.Sigmoid()
        
    def forward(self, data):
        x = self.relu(self.l1(data))
        x = self.dropout(x)
        x = self.l2(x)
        #x = self.out(x)
        return x

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc
'''
def CB_loss(labels, logits, samples_per_cls, no_of_classes, beta):
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(torch.tensor(labels.tolist()), no_of_classes).float().to(device) #F.one_hot(labels, no_of_classes).float()

    weights = torch.tensor(weights).float().to(device)
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1,no_of_classes)

    cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weight = weights)
    return cb_loss
'''
def data_loader(one_hot=False):
    X, y = make_classification(n_samples=1000, n_features=60,n_classes=2, n_clusters_per_class=2, weights=[0.95, 0.05], flip_y=0.1,  shuffle=True, random_state=1048)
    train_x, train_y = torch.from_numpy(X[:700]), torch.from_numpy(y[:700])
    val_x, val_y = torch.from_numpy(X[700:]), torch.from_numpy(y[700:])

    if one_hot:
        train_y = F.one_hot(torch.tensor(train_y.tolist()))
        val_y = F.one_hot(torch.tensor(val_y.tolist()))

    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=300, shuffle=True)
    
    return train_loader, val_loader

def train(model, train_loader, loss_type='default'):
    epochs = 40 
    lr = 0.01
    optimizer = torch.optim.SGD(model.parameters(),lr=lr)
    criterion = nn.CrossEntropyLoss()
    cb_criterion = CBLoss(2)

    model.train()
    for e in range(1, epochs+1):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.float().to(device)
            y_batch = y_batch.long().to(device)
            #y_batch = F.one_hot(torch.tensor(y_batch.tolist())).float().to(device)
            optimizer.zero_grad()
            
            y_pred = model(X_batch)
            
            _, samples_per_class = torch.unique(y_batch, return_counts=True)
            samples_per_class = samples_per_class.detach().cpu().numpy()
            if loss_type == 'cb-loss':
                loss = cb_criterion(y_batch, y_pred, samples_per_class)
            else:
                loss = criterion(y_pred , y_batch)

            output = torch.argmax(y_pred, dim=1)
            correct_results = (y_batch == output).sum().float()
            accuracy = correct_results/y_batch.size()[0]
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += accuracy.item()
            

        print(f'Epoch {e+0:02}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')

def eval(model, val_loader):
    model.eval()
    with torch.no_grad():
        for x,y in val_loader:
            x = x.float().to(device)
            y_test_pred = model(x)
            pred = torch.argmax(y_test_pred, dim=1)
            #pred = torch.round(y_test_pred)
            cf_matrix = confusion_matrix(y, pred.cpu().detach().numpy())
            print(f'Confusion matrix: \n {cf_matrix}')
            print(classification_report(y, pred.cpu().detach().numpy()))
            sns.heatmap(cf_matrix, annot=True)

# if __name__ == '__main__':
    '''no_of_classes = 2
    logits = torch.rand(10,no_of_classes).float()
    labels = torch.randint(0,no_of_classes, size = (10,))
    beta = 0.9999
    gamma = 2.0
    samples_per_cls = [9,1]
    loss_type = "sigmoid" '''
    #cb_loss = CB_loss(labels, logits, samples_per_cls, no_of_classes,loss_type, beta, gamma)
    # print(cb_loss)

    
    # print(f'--------------- Cross-Entropy -----------------')
    # model = MyNN().to(device)
    # train_loader, val_loader = data_loader()
    # train(model, train_loader)
    # eval(model, val_loader)

    # print(f'--------------- classBalance -----------------')
    # model_cb = MyNN().to(device)
    # train_loader, val_loader = data_loader()
    # train(model_cb, train_loader,'cb-loss')
    # eval(model_cb, val_loader)
    