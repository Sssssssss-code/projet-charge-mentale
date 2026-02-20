import torch
import torch.nn as nn
from torchvision import models
from load_data import train_freq_data_loader, valid_freq_data_loader, test_freq_data_loader, train_freq_data_loader_list, test_freq_data_loader_list
import matplotlib.pyplot as plt

models_list = []
optimizer_list = []
loss_list = []

def feature_extraction(y_batch,i) :
    res = []
    for sample in y_batch : 
        res.append(sample[i])
    return res


resnet = models.resnet18(weights='DEFAULT')
resnet.conv1 = nn.Conv2d(4,64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
resnet.fc = torch.nn.Linear(in_features=512, out_features=1)

for param in resnet.parameters():
    param.requires_grad = False

for param in resnet.fc.parameters():
	param.requires_grad = True

class FCNModel(nn.Module):
    def __init__(self):
        super().__init__()

        # separates all features in separate branches 
        self.branches = nn.ModuleList([resnet for _ in range(6)])


    def forward(self, x_batch, weight_batch):

        branch_outputs = []

        for branch in self.branches:
            branch_outputs.append(branch(x_batch))  

        branch_outputs = torch.stack(branch_outputs).float()
        branch_outputs = branch_outputs.permute(1, 0, 2)
        weights = weight_batch.float().unsqueeze(-1)
        weighted = (branch_outputs * weights).sum(dim=1) / weights.sum(dim=1)

        return weighted


def valid_epoch(test_loader, loss_func, model):
    model.eval()
    tot_loss, n_samples=0,0
    with torch.no_grad():
        for x_batch, y_batch, weight_batch in test_loader:

            preds = model(x_batch, weight_batch)

            loss = loss_func(preds.squeeze(), y_batch)
            
            n_samples += y_batch.size(0)
            tot_loss += loss.item() * y_batch.size(0)

    model.train()
    avg_loss = tot_loss / n_samples if n_samples > 0 else 0.0
    return avg_loss

def train_pretrained(train_freq_data_loader, valid_freq_data_loader, optimizer, loss_fn, n_epochs, model, verb=1):
    
    train_loss_list = []
    valid_loss_list = []
    
    epoch_loss = 0
    n_samples = 0

    for epoch in range(n_epochs):
        for x_batch, y_batch, weight_batch in train_freq_data_loader:
            model.train()
            optimizer.zero_grad()
            outputs = model(x_batch, weight_batch)
            loss = loss_fn(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * y_batch.size(0)
            n_samples += y_batch.size(0)
        
        with torch.no_grad():
            valid_loss = valid_epoch(valid_freq_data_loader, loss_fn, model)
            valid_loss_list.append(valid_loss)
        
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {(epoch_loss/n_samples):.4f}, Valid loss: {valid_loss:.4f}")
        train_loss_list.append(epoch_loss/n_samples)
        
    if verb:
        plt.plot(range(len(train_loss_list)), train_loss_list, label='train')
        plt.plot(range(len(valid_loss_list)), valid_loss_list, label='valid')
            
        print(f'test mse: {valid_epoch(test_freq_data_loader, loss_fn, model)}')
        print(f'test mae: {valid_epoch(test_freq_data_loader, nn.L1Loss(), model)}')

        plt.legend()
        plt.show()
    
    return valid_loss
        
def train_lopo_pretrained(train_freq_data_loader_list, valid_freq_data_loader_list, n_epochs, verb=1):
    if len(train_freq_data_loader_list) != len(valid_freq_data_loader_list):
        raise TabError(f"different size for train_res_data_loader_list and valid_res_data_loader_list\n\
                        got {len(train_freq_data_loader_list)} and {len(valid_freq_data_loader_list)}")
    else:
        valid_loss_list = []
        for i in range(len(train_freq_data_loader_list)):
            model = FCNModel()
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            train_data_loader = train_freq_data_loader_list[i]
            test_data_loader = valid_freq_data_loader_list[i]
            valid_loss = train_pretrained(train_freq_data_loader=train_data_loader,
                                          valid_freq_data_loader=test_data_loader,
                                          optimizer=optimizer,
                                          loss_fn=loss_fn,
                                          n_epochs=n_epochs,
                                          model=model,
                                          verb=verb)
            valid_loss_list.append(valid_loss)
            print(f'valid_loss: {valid_loss}')
        plt.scatter(list(range(len(train_freq_data_loader_list))), valid_loss_list)
        plt.show()


if __name__ == "__main__":
    
    n_epochs = 50
        
    model = FCNModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_lopo_pretrained(train_freq_data_loader_list, test_freq_data_loader_list, n_epochs=20, verb=0)

