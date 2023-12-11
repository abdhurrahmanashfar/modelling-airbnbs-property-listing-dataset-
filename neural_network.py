import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import yaml
import os
import json
import time
import datetime
import torch.optim as optim
from pandasgui import show
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tabular_df_alt import load_airbnb
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
torch.manual_seed(2)

class AirbnbNightlyPriceRegressionDataset(Dataset):
    def __init__(self, features, label):
        super().__init__()
        
        scalar = StandardScaler()
        self.features = features
        self.label = label

        scalar = scalar.fit(features)
        # Transform your features using the fitted scaler
        self.features = pd.DataFrame(scalar.transform(features))

        # Or alt way to use scalar:
        # self.features = pd.DataFrame(scalar.fit_transform(self.features))

    def __getitem__(self, idx):
        return torch.tensor(self.features.iloc[idx]).float(), torch.tensor(self.label.iloc[idx]).float()
        # features = self.features.iloc[idx]
        # features = torch.tensor(features).float()
        # label = self.label.iloc[idx]
        # label = torch.tensor(label).float()
        # return features, label
        
    def __len__(self):
        return len(self.features)


def get_random_split(dataset):
    train_set, placeholder_set = random_split(dataset, [0.7, 0.3])
    test_set, validation_set = random_split(placeholder_set, [0.5, 0.5])
    
    return train_set, validation_set, test_set


class TabularNN(nn.Module):
    def __init__(self, configs):
        super().__init__()
        hidden_layer_width = configs["hidden_layer_width"]
        depth = configs["depth"]

        # Create a list of layers
        layers = []

        # Input layer
        layers.append(nn.Linear(11, hidden_layer_width))
        layers.append(nn.ReLU())
        # layers.append(nn.BatchNorm1d(hidden_layer_width))  # BatchNorm after ReLU
        # layers.append(nn.Dropout(p=0.2))

        for _ in range(depth - 1):
            # Hidden layers
            layers.append(nn.Linear(hidden_layer_width, hidden_layer_width))
            layers.append(nn.ReLU())
            #layers.append(nn.BatchNorm1d(hidden_layer_width))  # BatchNorm after ReLU
            layers.append(nn.Dropout(p=0.2))

        # Output layer
        layers.append(nn.Linear(hidden_layer_width, 1))

        self.model = nn.Sequential(*layers)
      
    def forward(self, x):
        output = self.model(x)
        return output
    

def generate_nn_configs():
    configs = []

    hidden_layer_widths = [10, 20]
    depths = [1, 2]
    learning_rates = [0.001, 0.01]
    optimisers = ['SGD']

    for config in itertools.product(hidden_layer_widths, depths, learning_rates, optimisers):
        print(config)
        config_dict = {
            'hidden_layer_width': config[0],
            'depth': config[1],
            'learning_rate': config[2],
            'optimiser': config[3]
        }
        configs.append(config_dict)

    return configs


def get_nn_config():
    with open('nn_config.yaml', 'r') as file:
        configs = yaml.safe_load(file)
    return configs


def train(model, train_loader, val_loader, num_epochs, config):
    batch_idx = 0
    learning_rate = config['learning_rate']
    optimiser = optim.Adam(model.parameters(), learning_rate)
    writer = SummaryWriter()
    
    train_rmse_batch = []
    val_rmse_list = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for features, label in train_loader:
            prediction = model(features)

            #making label a 2d tensor
            label = torch.unsqueeze(label, 1)
           # print(label.shape)

            loss = F.mse_loss(prediction, label)
            label = label.detach().numpy()
            prediction = prediction.detach().numpy()

            R2_train = r2_score(label, prediction)
            RMSE_train = torch.sqrt(loss)

            print('train')
            print(RMSE_train.item())

            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

            writer.add_scalar("loss", loss.item(), batch_idx)
            batch_idx += 1
            running_loss += loss.item()
            train_rmse_batch.append(RMSE_train.item())


        # Validation phase
        model.eval()
        # val_loss = 0.0
        with torch.no_grad():
            for features, label in val_loader:
                prediction = model(features)
                #making label a 2d tensor
                label = torch.unsqueeze(label, 1)

                loss = F.mse_loss(prediction, label)
                #label = label.view(-1, 1)

                # val_loss += loss.item()
                R2_val = r2_score(label.detach().numpy(), prediction.detach().numpy())
                # R2_val = R2_val(prediction, label)
                RMSE_val = torch.sqrt(loss)
                val_rmse_list.append(RMSE_val)
    
        avg_train_loss = float(np.average(train_rmse_batch))
        avg_val_loss = float(np.average(val_rmse_list))
        print('val')
        print(avg_val_loss)

        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    metrics = {
        "RMSE_train": avg_train_loss, 
        "R2_train": R2_train, 
        "R2_val": R2_val, 
        "RMSE_val": avg_val_loss,
    }
    return metrics


def save_model(model, hyperparameters, metrics, save_folder):
    os.makedirs(save_folder, exist_ok=True)

    if isinstance(model, torch.nn.Module):
        torch.save(model.state_dict(), os.path.join(save_folder, "model.pt"))

    with open(os.path.join(save_folder, "hyperparameters.json"), "w") as f:
        json.dump(hyperparameters, f)

    with open(os.path.join(save_folder, "metrics.json"), "w") as f:
        json.dump(metrics, f)


def find_best_nn():
    num_epochs = 100

    best_model = None
    best_metrics = None
    best_hyperparameters = None
    best_config = None
    best_loss = 10000000

    configs = generate_nn_configs()
    configs_not_working = []
    for i, config in enumerate(configs):
        print(f"Training Model {i+1}...")
        print(config)
        model = TabularNN(config)
        
        # try:
        #     metrics = train(model, train_loader, validation_loader, num_epochs, config)
        # except:
        #     print(config)
        #     configs_not_working.append(config)


    #     #########################################################################################
    #     # Use this instead of the try except to see the errors "Input contains Nan"
        metrics = train(model, train_loader, validation_loader, num_epochs, config)

        if metrics['RMSE_val'] < best_loss:
            best_model = model
            best_metrics = metrics
            best_hyperparameters = config
            # best_config = config
            best_loss = metrics['RMSE_val']

    #     # save_config(config, f'model_{i+1}')
    save_best_model_folder = "models/neural_networks/regression/best_model"
    save_model(best_model, best_hyperparameters, best_metrics, save_best_model_folder)
    # print(configs_not_working)


    return model, best_model, best_metrics, best_hyperparameters, best_config


if __name__ == "__main__":
    df = pd.read_csv("D:/AiCore/Projects/AirBnb/airbnb-property-listings/tabular_data/Cleaned_AirBnbData_3.csv")

    features, label = load_airbnb(df, 'Price_Night')
    del features["Unnamed: 0"]
    features = features.select_dtypes(include=["int64", "float64"])

    # show(features)
    # show(label)
    batch_size = 8


    dataset = AirbnbNightlyPriceRegressionDataset(features, label)

    train_set, validation_set, test_set = get_random_split(dataset)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


    configs = generate_nn_configs()
    metrics = []
    best_loss = 10000000
    for i, config in enumerate(configs):
        print(config)
        model = TabularNN(config)
        metrics.append(train(model, train_loader, validation_loader, 100, config))
        if metrics[i]['RMSE_val'] < best_loss:
            best_model = model
            best_metrics = metrics[i]
            best_hyperparameters = config
            best_config = config
            best_loss = metrics[i]['RMSE_val']

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_folder = f"models/neural_networks/regression/{current_time}"
    save_model(model, configs, metrics, save_folder)

    print(f"Training Complete")
    print(f"Best Model Configuration: {best_config}")
    print(f"Best Metrics: {best_metrics}")
    print(f"Best Loss: {best_loss}")
    
