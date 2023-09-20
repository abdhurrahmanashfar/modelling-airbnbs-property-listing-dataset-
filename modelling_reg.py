# modelling.py

import os
import joblib
import json
import pandas as pd
import numpy as np
import glob
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from itertools import product
from tabular_df import load_airbnb


df = pd.read_csv("D:/AiCore/Projects/AirBnb/airbnb-property-listings/tabular_data/Cleaned_AirBnbData.csv")

np.random.seed(2)
X, y = load_airbnb(df, "Price_Night")
X = X.select_dtypes(include="number")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1)

data_sets = [X_train, y_train, X_test, y_test, X_val, y_val]


# Train the linear regression model

# model = SGDRegressor(random_state=42)
# model = DecisionTreeRegressor(max_depth=1)
# model = RandomForestRegressor(max_depth=1)
model = GradientBoostingRegressor(max_depth=1)

model.fit(X_train, y_train)

# Make predictions on both training and test data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate the model
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("Mean Squared Error - Training:", mse_train)
print("R-squared - Training:", r2_train)
print("Root Mean Squared Error - Training:", rmse_train)

print("Mean Squared Error - Test:", mse_test)
print("R-squared - Test:", r2_test)
print("Root Mean Squared Error - Test:", rmse_test)


# hyperparameters = {
#     'fit_intercept': [True, False],
#     'alpha' : [0.001],
#     'early_stopping' :[True, False],
#     'loss' : ["squared_error"],
#     'learning_rate' : ["optimal"]
# }

# #DecisionTreeRegressor
# hyperparameters = {
#     'max_depth': [1]
#     }

# #RandomForestRegressor 
# hyperparameters = {
#     'n_estimators': [50, 100, 200], 'max_depth': [1]
#     }

#GradientBoostingRegressor
hyperparameters = {
    'n_estimators': [50, 100, 200], 'max_depth': [1]
    }


def custom_tune_regression_model_hyperparameters(model_class, X_train, y_train, X_val, y_val, X_test, y_test, hyperparameters):
    best_model = None
    best_params = None
    best_val_rmse = float('inf')
    
    for param_values in product(*hyperparameters.values()):
        params = dict(zip(hyperparameters.keys(), param_values))
        model = SGDRegressor(random_state=42)
        model.fit(X_train, y_train)
        
        # Calculate validation RMSE
        val_pred = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model = model
            best_params = params
    
    # Calculate test RMSE using the best model
    test_pred = best_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    performance_metrics = {
        "validation_RMSE": best_val_rmse,
        "test_RMSE": test_rmse
    }
    
    return best_model, best_params, performance_metrics



# best_model, best_params, performance_metrics = custom_tune_regression_model_hyperparameters(
#     model_class=LinearRegression,
#     X_train=X_train, y_train=y_train,
#     X_val=X_val, y_val=y_val,
#     X_test=X_test, y_test=y_test,
#     hyperparameters=hyperparameters
# )

# print("")
# print("Best Hyperparameters:", best_params)
# print("Validation RMSE:", performance_metrics["validation_RMSE"])
# print("Test RMSE:", performance_metrics["test_RMSE"])



def tune_regression_model_hyperparameters(model, data_sets, hyperparameters):    
    model = model
    gs_reg = GridSearchCV(model, hyperparameters, verbose=10, refit=True)
    gs_reg.fit(data_sets[0], data_sets[1])
    best_model = gs_reg.best_estimator_
    best_iteration_hyperparams = gs_reg.best_params_
    y_pred = model.predict(X)
    RMSE = mean_squared_error(y, y_pred, squared=False)
    MAE = mean_absolute_error(y, y_pred)
    R2 = r2_score(y, y_pred)
    # metrics = [
    #     {"RMSE" : RMSE},
    #     {"MAE" : MAE},
    #     {"R2" : R2}
    #     ]           
    return RMSE, MAE, R2, best_iteration_hyperparams, best_model

RMSE, MAE, R2, best_iteration_hyperparams, best_model = tune_regression_model_hyperparameters(model, data_sets, hyperparameters)
print(RMSE)
print(MAE)
print(R2)
print(best_iteration_hyperparams)
print(best_model)

performance_metrics = [
    {"RMSE" : RMSE},
    {"MAE" : MAE},
    {"R2" : R2}
    ] 


def evaluate_all_models(X, y):
        # Tune hyperparameters
        RMSE, MAE, R2, best_iteration_hyperparams, best_model = tune_regression_model_hyperparameters(model, data_sets, hyperparameters)

# if __name__ == "__main__":
#     evaluate_all_models(X, y)


def save_model(model, hyperparameters, performance_metrics, folder):
    # Create the folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    
    # Save the trained model
    model_filename = os.path.join(folder, "model.joblib")
    joblib.dump(model, model_filename)
    
    # Save hyperparameters as JSON
    hyperparameters_filename = os.path.join(folder, "hyperparameters.json")
    with open(hyperparameters_filename, "w") as f:
        json.dump(hyperparameters, f)
    
    # Save performance metrics as JSON
    metrics_filename = os.path.join(folder, "metrics.json")
    with open(metrics_filename, "w") as f:
        json.dump(performance_metrics, f)


# folder_to_save = "models/regression/linear_regression"
# folder_to_save = "models/regression/decision_tree"
# folder_to_save = "models/regression/randomforest"
folder_to_save = "models/regression/gradientboosting"

# save_model(best_model, best_params, performance_metrics, folder=folder_to_save)

save_models = save_model(best_model, best_iteration_hyperparams, performance_metrics, folder=folder_to_save)
# print(save_models)
print("Model, hyperparameters, and metrics saved successfully.")




def find_best_model(reg):     
    metrics_files = glob.glob("./models/regression/*/metrics.json", recursive=True)
    score_list = []
    
    for file in metrics_files:
        
        f = open(str(file))
        dic_metrics = json.load(f)
        score = dic_metrics[0]
        
        rmse = score["RMSE"]
        score_list.append(rmse)
        # print(score_list)
        # if rmse < best_score:
            # best_score = rmse
            # best_score = min(score_list)
            # best_name = str(file).split('\\')[1]
    best_score = min(score_list)
    best_name = str(file).split('\\')[1]
    
    path = f'./models/regression/{best_name}/'
    # model = joblib.load(path + 'model.joblib')
    
    with open (path + 'hyperparameters.json', 'r') as fp:
        param = json.load(fp)
    
    with open (path + 'metrics.json', 'r') as fp:
        metrics = json.load(fp)
    
    return model, param, metrics



if __name__ == "__main__":
    evaluate_all_models(X, y)
    model, param, metrics = find_best_model("regression")
    print ('best classification model is: ', model)
    print('with metrics', metrics)    




