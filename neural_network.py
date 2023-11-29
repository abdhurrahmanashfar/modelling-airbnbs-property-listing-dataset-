
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
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tabular_df import load_airbnb
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
torch.manual_seed(2)
df = pd.read_csv("D:/AiCore/Projects/AirBnb/airbnb-property-listings/tabular_data/Cleaned_AirBnbData.csv")

class AirbnbNightlyPriceRegressionDataset(Dataset):
    def __init__(self, features, label):
        super().__init__()
        # self.features = torch.tensor(features, dtype=torch.float32)
        # self.prices = torch.tensor(prices, dtype=torch.float32)
        # scaler = StandardScaler()
        # self.features = torch.tensor(scaler.fit_transform(self.features), dtype=torch.float32)
        self.features, self.label = load_airbnb(df, 'Price_Night')
        self.features = self.features.select_dtypes(include=["int64", "float64"])
        scaler = StandardScaler()
        # self.features = torch.tensor(scaler.fit_transform(self.features), dtype=torch.float32)
        # self.features = scaler.fit_transform(self.features)
        self.features = pd.DataFrame(scaler.fit_transform(self.features), columns=self.features.columns)

    def __getitem__(self, idx):
        # return torch.tensor(self.features.iloc[idx]).float(), torch.tensor(self.label.iloc[idx]).float()
        features = self.features.iloc[idx]
        features = torch.tensor(features).float()
        label = self.label.iloc[idx]
        label = torch.tensor(label).float()
        return features, label
        
    def __len__(self):
        return len(self.features)

# features, label = load_airbnb(df, 'Price_Night')
# del features["Unnamed: 0"]
# dataset = AirbnbNightlyPriceRegressionDataset(features, label)
original_df = pd.read_csv("D:/AiCore/Projects/AirBnb/airbnb-property-listings/tabular_data/Cleaned_AirBnbData_2.csv")
df['bedrooms'] = original_df['bedrooms']
print(df.isnull().sum())


def get_random_split(dataset):
    train_set, test_set = random_split(dataset, [0.7, 0.3])
    train_set, validation_set = random_split(train_set, [0.5, 0.5])
    
    return train_set, validation_set, test_set

batch_size = 64

# train_set, validation_set, test_set = get_random_split(dataset)
# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
# validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
# # data_loader = {"train": train_loader, "validation": validation_loader, "test": test_loader}


class TabularNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, configs):
        # super(TabularNN, self).__init__()
        super().__init__()
        # self.fc1 = nn.Linear(input_size, configs['hidden_layer_width'])
        # self.bn1 = nn.BatchNorm1d(input_size)
        # self.relu = nn.ReLU()
        # self.m = nn.Dropout(p=0.2)
        # self.n = nn.Sequential()
        # self.fc2 = nn.Linear(configs['hidden_layer_width'], output_size)
        # self.optimizer = getattr(optim, configs['optimiser'])(self.parameters(), lr=configs['learning_rate'])

        layers = []
        layers.append(nn.Linear(input_size, configs['hidden_layer_width']))
        # layers.append(nn.BatchNorm1d(input_size))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(configs['hidden_layer_width'], configs['hidden_layer_width']))
        layers.append(nn.Linear(configs['hidden_layer_width'], output_size))
        self.layers = nn.Sequential(*layers)
        # self.optimizer = getattr(optim, configs['optimiser'])(self.parameters(), lr=configs['learning_rate'])
        # self.optimiser = torch.optim.SGD(model.parameters(), configs['learning_rate'])
        self.optimiser = torch.optim.SGD(self.parameters(), configs['learning_rate'])

    def forward(self, x):
        # out = self.fc1(x)
        # out = self.relu(out)
        # out = self.fc2(out)
        # return out
        # return self.layers(features)
        x = self.layers(x)
        return x
    

# output size (in regression tasks it's 1).
# input_size = 10
# hidden_size = 64
# output_size = 1
# input_size = 10
# hidden_size = 64
# output_size = 1


def generate_nn_configs():
    configs = []

    hidden_layer_widths = [32, 64, 128]
    depths = [2, 3, 4]
    learning_rates = [0.001, 0.01, 0.1]
    # optimisers = ['Adadelta', 'SGD', 'Adam', 'Adagrad']
    optimisers = ['SGD']
    # hidden_layer_widths = [32]
    # depths = [2]
    # learning_rates = [0.001, 0.01, 0.1]
    # optimisers = ['SGD', 'Adam']

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


# configs = generate_nn_configs()

# file_path = 'nn_config.yaml'

# with open('nn_config.yaml', 'w') as file:
#     yaml.dump(configs, file)


def get_nn_config():
    with open('nn_config.yaml', 'r') as file:
        configs = yaml.safe_load(file)
    return configs

# saved_configs = get_nn_config()

# model = TabularNN(input_size, hidden_size, output_size, saved_configs)

def train(model, train_loader, val_loader, num_epochs):
    batch_idx = 0
    pred_time = []
    start_time = time.time()
    optimiser = model.optimiser
    criterion = nn.MSELoss()
    writer = SummaryWriter()

    # optimiser_name = config['optimiser']
    # learning_rate = config['learning_rate']

    # optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for features, label in train_loader:
            # label = torch.unsqueeze(label, 1)
            time_b4_pred = time.time()
            prediction = model(features)
            time_after_pred = time.time()
            time_elapsed = time_after_pred - time_b4_pred
            pred_time.append(time_elapsed)
            print(prediction.shape)
            loss = criterion(prediction, label)
            # prediction[torch.isnan(prediction)] = 1  # Replace NaN values with 1
            # label[torch.isnan(label)] = 1  # Replace NaN values with 1
            # R2_train = r2_score(label.detach().numpy(), prediction.detach().numpy())
            R2_train = r2_score(prediction.detach().numpy(), label.detach().numpy())
            # R2_train = R2_train(prediction, label)
            RMSE_train = torch.sqrt(loss)
            loss.backward()
            print(loss.item())
            optimiser.step()
            optimiser.zero_grad()
            writer.add_scalar("loss", loss.item(), batch_idx)
            batch_idx += 1
            running_loss += loss.item()

        end_time = time.time()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, label in val_loader:
                prediction = model(features)
                loss = criterion(prediction, label)
                val_loss += loss.item()
                # R2_val = r2_score(label.detach().numpy(), prediction.detach().numpy())
                R2_val = r2_score(prediction.detach().numpy(), label.detach().numpy())
                # R2_val = R2_val(prediction, label)
                RMSE_val = torch.sqrt(loss)
        
        # Calculate average losses
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        # avg_test_loss = test_loss / len(test_loader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # test_loss = evaluate_model(model, criterion, test_loader, phase='Test')

    training_duration = end_time - start_time
    inference_latency = sum(pred_time)/len(pred_time)

    metrics = {
        "RMSE_train": RMSE_train, 
        "R2_train": R2_train, 
        "training_duration": training_duration, 
        "inference_latency": inference_latency, 
        "R2_val": R2_val, 
        "RMSE_val": RMSE_val,
        "val_loss": val_loss
    }

    # for key, value in metrics.items():
    #     print(f'The type of {key} is {type(value)}')

    keys_to_convert = ["RMSE_train", "RMSE_val"]

    for key in keys_to_convert:
        # metrics[key] = np.array(metrics[key])
        # metrics[key] = metrics[key].detach().numpy()
        metrics[key] = metrics[key].item()

    # return RMSE_train, R2_train, training_duration, inference_latency, R2_val, RMSE_val, val_loss
    return metrics


num_epochs = 15


def save_model(model, hyperparameters, metrics, save_folder):
    os.makedirs(save_folder, exist_ok=True)

    if isinstance(model, torch.nn.Module):
        torch.save(model.state_dict(), os.path.join(save_folder, "model.pt"))

    with open(os.path.join(save_folder, "hyperparameters.json"), "w") as f:
        json.dump(hyperparameters, f)

    with open(os.path.join(save_folder, "metrics.json"), "w") as f:
        json.dump(metrics, f)

# current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# save_folder = f"models/neural_networks/regression/{current_time}"

# save_best_model_folder = "models/neural_networks/regression/best_model"


def find_best_nn():
    best_model = None
    best_metrics = None
    best_hyperparameters = None
    best_config = None
    best_loss = 10000

    configs = generate_nn_configs()

    for i, config in enumerate(configs):
        print(f"Training Model {i+1}...")
        model = TabularNN(input_size, hidden_size, output_size, config)
        
        # try:
        metrics = train(model, train_loader, validation_loader, num_epochs)
        # except:
        #     print(config)
        #     break

        if metrics['val_loss'] < best_loss:
            best_model = model
            best_metrics = metrics
            best_hyperparameters = config
            # best_config = config
            best_loss = metrics['val_loss']

        # save_config(config, f'model_{i+1}')
    print(type(best_metrics))
    save_model(best_model, best_hyperparameters, best_metrics, save_best_model_folder)

    return model, best_model, best_metrics, best_hyperparameters, best_config


if __name__ == "__main__":
    # model, best_model, best_metrics, best_hyperparameters, best_config = find_best_nn()
    # # optimiser = model.optimizer
    # metrics = train(model, train_loader, validation_loader, num_epochs)
    # save_model(model, saved_configs, metrics, save_folder)
    # print(f"Best Model Configuration: {best_config}")
    # print(f"Best Metrics: {best_metrics}")
    # print(f"Best Hyperparameters: {best_hyperparameters}")

    features, label = load_airbnb(df, 'Price_Night')
    del features["Unnamed: 0"]
    dataset = AirbnbNightlyPriceRegressionDataset(features, label)

    # print(features.isnull().sum())
    # print(label.isnull().sum())

    train_set, validation_set, test_set = get_random_split(dataset)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    # data_loader = {"train": train_loader, "validation": validation_loader, "test": test_loader}

    input_size = 10
    hidden_size = 64
    output_size = 1

    configs = generate_nn_configs()

    with open('nn_config.yaml', 'w') as file:
        yaml.dump(configs, file)

    saved_configs = get_nn_config()

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_folder = f"models/neural_networks/regression/{current_time}"

    save_best_model_folder = "models/neural_networks/regression/best_model"

    model, best_model, best_metrics, best_hyperparameters, best_config = find_best_nn()
    # optimiser = model.optimizer
    # torch.autograd.set_detect_anomaly(True)
    metrics = train(model, train_loader, validation_loader, num_epochs)
    # torch.autograd.set_detect_anomaly(False)
    save_model(model, saved_configs, metrics, save_folder)
    print(f"Best Model Configuration: {best_config}")
    print(f"Best Metrics: {best_metrics}")
    print(f"Best Hyperparameters: {best_hyperparameters}")







