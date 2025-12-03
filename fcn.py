import torch
import torch.nn as nn
import torch.optim as optim


class FCN(nn.Module):
    
    def __init__(self, ecg_input_size, gsr_input_size, inf_ppg_input_size, pix_ppg_input_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.conv_ecg = nn.Conv1d(ecg_input_size, 128)
        self.conv_gsr = nn.Conv1d(gsr_input_size, 128)
        self.conv_inf_ppg = nn.Conv1d(inf_ppg_input_size, 128)
        self.conv_pix_ppg = nn.Conv1d(pix_ppg_input_size, 128)
        self.conv2 = nn.Conv1d(128, 128)
        self.gap = nn.AvgPool1d()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        
    def forward(self, x_ecg, x_gsr, x_inf_ppg, x_pix_ppg):
        
        # ecg layer
        x_ecg = self.relu(self.conv_ecg(x_ecg))
        x_ecg = self.relu(self.conv2(x_ecg))
        x_ecg = self.relu(self.conv2(x_ecg))
        x_ecg = self.gap(x_ecg)
        
        # gsr layer
        x_gsr = self.relu(self.conv_gsr(x_gsr))
        x_gsr = self.relu(self.conv2(x_gsr))
        x_gsr = self.relu(self.conv2(x_gsr))
        x_gsr = self.gap(x_gsr)
        
        # inf ppg layer
        x_inf_ppg = self.relu(self.conv_inf_ppg(x_inf_ppg))
        x_inf_ppg = self.relu(self.conv2(x_inf_ppg))
        x_inf_ppg = self.relu(self.conv2(x_inf_ppg))
        x_inf_ppg = self.gap(x_inf_ppg)
        
        # pix ppg layer
        x_pix_ppg = self.relu(self.conv_pix_ppg(x_pix_ppg))
        x_pix_ppg = self.relu(self.conv2(x_pix_ppg))
        x_pix_ppg = self.relu(self.conv2(x_pix_ppg))
        x_pix_ppg = self.gap(x_pix_ppg)
        
        # concatenate all layers
        x = torch.cat(x_ecg, x_gsr, x_inf_ppg, x_pix_ppg, dim=1)
        x = self.softmax(self.relu(x))
        
        return x
        

    
