import json
import numpy as np
import os
import sys
import torch
from datasets import mnist, celeba
from neural_process import NeuralProcessImg
from time import strftime
from training import NeuralProcessTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get config file from command line arguments
if len(sys.argv) != 2:
    raise(RuntimeError("Wrong arguments, use python main_experiment.py <path_to_config>"))
config_path = sys.argv[1]

# Create a folder to store experiment results
timestamp = strftime("%Y-%m-%d_%H-%M")
directory = "results_{}".format(timestamp)
if not os.path.exists(directory):
    os.makedirs(directory)

# Open config file
with open(config_path) as config_file:
    config = json.load(config_file)

# Save config file in experiment directory
with open(directory + '/config.json', 'w') as config_file:
    json.dump(config, config_file)

img_size = config["img_size"]
batch_size = config["batch_size"]
r_dim = config["r_dim"]
h_dim = config["h_dim"]
z_dim = config["z_dim"]
num_context_range = config["num_context_range"]
num_extra_target_range = config["num_extra_target_range"]
epochs = config["epochs"]

if config["dataset"] == "mnist":
    data_loader, _ = mnist(batch_size=batch_size, size=img_size[1])
elif config["dataset"] == "celeba":
    data_loader = celeba(batch_size=batch_size, size=img_size[1])

np_img = NeuralProcessImg(img_size, r_dim, z_dim, h_dim).to(device)

optimizer = torch.optim.Adam(np_img.parameters(), lr=config["lr"])

np_trainer = NeuralProcessTrainer(device, np_img, optimizer,
                                  num_context_range, num_extra_target_range,
                                  print_freq=100)

for epoch in range(epochs):
    print("Epoch {}".format(epoch + 1))
    np_trainer.train(data_loader, 1)
    # Save losses at every epoch
    with open(directory + '/losses.json', 'w') as f:
        json.dump(np_trainer.epoch_loss_history, f)
    # Save model at every epoch
    torch.save(np_trainer.neural_process.state_dict(), directory + '/model.pt')
