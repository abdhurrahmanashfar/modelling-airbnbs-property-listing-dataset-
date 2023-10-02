
import os
import joblib
import json
import glob
import pandas as pd
import numpy as np
from tabular_df import load_airbnb
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# from modelling_reg import save_model


df = pd.read_csv("D:/AiCore/Projects/AirBnb/airbnb-property-listings/tabular_data/Cleaned_AirBnbData_2.csv")

df = df.drop(['Unnamed: 0', 'ID', 'Title', 'Description', 'Amenities','Location', 'guests','url', 'bedrooms'], axis=1)

X, y = load_airbnb(df, "Category")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

data_sets = [X_train, y_train, X_test, y_test, X_val, y_val]

# Initialize and train the Logistic Regression model
# model = LogisticRegression(max_iter=10000)
# model = DecisionTreeClassifier(max_depth=2)
# model = RandomForestClassifier(n_estimators=50, max_depth=10)
model = GradientBoostingClassifier(n_estimators=50, max_depth=3)

model.fit(X_train, y_train)

# Predict on the training set
y_train_pred = model.predict(X_train)

# Predict on the test set
y_test_pred = model.predict(X_test)

# Calculate accuracy_score
accuracy = accuracy_score(y_test, y_test_pred)
print(f"Accuracy: {accuracy}")

# Calculate precision_score
precision = precision_score(y_test, y_test_pred, average="macro")
print(f"Precision: {precision}")

# Calculate recall_score
recall = recall_score(y_test, y_test_pred, average="macro")
print(f"Recall: {recall}")

# Calculate f1_score
f1 = f1_score(y_test, y_test_pred, average="macro")
print(f"F1-Score: {f1}")

# Calculate confusion_matrix
cm = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:")
print(cm)


performance_metrics = [
    {"Accuracy" : accuracy},
    {"Precision" : precision},
    {"Recall" : recall},
    {"F1-Score" : f1},
    # {"Confusion Matrix" : cm}
] 

# # LogisticRegression
# hyperparameters = {
#     'max_iter': [10000],
#     "random_state" : [42]
#     }

# # DecisionTreeClassifier
# hyperparameters = {
#     'max_depth': [2]
#     }

# # RandomForestClassifier
# hyperparameters = {
#     'n_estimators': [50],
#     'max_depth': [10]
#     }

# GradientBoostingClassifier
hyperparameters = {
    'n_estimators': [50],
    'max_depth': [3]
    }


# Calculate metrics for training set
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred, average='weighted')
train_recall = recall_score(y_train, y_train_pred, average='weighted')
train_f1_score = f1_score(y_train, y_train_pred, average='weighted')

# Calculate metrics for test set
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')
test_f1_score = f1_score(y_test, y_test_pred, average='weighted')

# Print the metrics
print("\n")
print("Training Set Metrics:")
print(f"Accuracy: {train_accuracy}")
print(f"Precision: {train_precision}")
print(f"Recall: {train_recall}")
print(f"F1 Score: {train_f1_score}")

print("\nTest Set Metrics:")
print(f"Accuracy: {test_accuracy}")
print(f"Precision: {test_precision}")
print(f"Recall: {test_recall}")
print(f"F1 Score: {test_f1_score}")


def tune_classification_model_hyperparameters(model, data_sets, hyperparameters):
    model = model
    grid_search = GridSearchCV(model, hyperparameters, cv=5, n_jobs=-1)
    grid_search.fit(data_sets[0], data_sets[1])
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Calculate validation accuracy
    y_val_pred = best_model.predict(X_val)
    validation_accuracy = accuracy_score(y_val, y_val_pred)

    # Calculate test accuracy
    y_test_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    return best_model, best_params, validation_accuracy, test_accuracy

best_model, best_params, validation_accuracy, test_accuracy = tune_classification_model_hyperparameters(model, data_sets, hyperparameters)
print(best_model)
print(best_params)
print({"validation_accuracy": validation_accuracy, "test_accuracy": test_accuracy})

# folder_to_save = "models/classification/logisticregression"
# folder_to_save = "models/classification/decisiontreeclassifier"
# folder_to_save = "models/classification/randomforestclassifier"
folder_to_save = "models/classification/gradientboostingclassifier"


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


save_models = save_model(best_model, best_params, performance_metrics, folder=folder_to_save)
print("Model, hyperparameters, and metrics saved successfully.")

def evaluate_all_models(X, y):
    best_model, best_params, validation_accuracy, test_accuracy = tune_classification_model_hyperparameters(model, data_sets, hyperparameters)

def find_best_model(reg):     
    metrics_files = glob.glob("./models/classification/*/metrics.json", recursive=True)
    score_list = []
    
    for file in metrics_files:
        
        f = open(str(file))
        dic_metrics = json.load(f)
        score = dic_metrics[3]
        
        f1 = score["F1-Score"]
        score_list.append(f1)

    best_score = min(score_list)
    best_name = str(file).split('\\')[1]
    
    path = f'./models/classification/{best_name}/'
    # model = joblib.load(path + 'model.joblib')
    
    with open (path + 'hyperparameters.json', 'r') as fp:
        param = json.load(fp)
    
    with open (path + 'metrics.json', 'r') as fp:
        metrics = json.load(fp)
    
    return model, param, metrics


if __name__ == "__main__":
    evaluate_all_models(X, y)
    model, param, metrics = find_best_model("classification")
    print ('best classification model is: ', model)
    print('with metrics', metrics)




