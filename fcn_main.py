from fcn import FCNModel, train, test_model, train_lopo_FCN
import matplotlib.pyplot as plt
import torch
import numpy as np
import rnn
import load_data


fcn_net = FCNModel(num_signals=3, kernel_size=7)
# rnn_net = rnn.MultiSignalRNN(num_signals=4)

loss_func = torch.nn.MSELoss()
mae_loss = torch.nn.L1Loss()
optim_adam = torch.optim.Adam(params= fcn_net.parameters())

last_valid_loss_list=train_lopo_FCN(train_data_loader_list=load_data.train_data_loader_list,
               valid_data_loader_list=load_data.test_data_loader_list,
               n_epochs=5)

# # train_loss_list, valid_loss_list = train(fcn_net, load_data.train_data_loader, load_data.valid_data_loader, loss_func, optim_adam, n_epochs=50)
# plt.plot(range(len(train_loss_list)), train_loss_list, label='train')

# print(f'test loss (mse): {test_model(fcn_net, load_data.test_data_loader, loss_func=loss_func)}')
# print(f'test loss (mae): {test_model(fcn_net, load_data.test_data_loader, loss_func=mae_loss)}')
print(f'Last validation losses: {np.array(last_valid_loss_list)}')
print(f"Mean: {np.array(last_valid_loss_list).mean()}")

print(f"Var: {np.array(last_valid_loss_list).var()}")
plt.plot(range(len(last_valid_loss_list)), last_valid_loss_list, label='valid')
plt.legend()
plt.show()

