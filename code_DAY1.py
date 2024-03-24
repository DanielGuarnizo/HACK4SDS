import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import KFold

train_dataset = pd.read_csv("HACK4SDS/Dataset_DAY1/Data/train_set.csv", delimiter=';')

## drop features
def Drop_unneed_columns(test, dataset):
    cols= ['days_to_default', 'application_ID', 'decision_date', 'company_ID']
    if test:
        cols.remove('days_to_default')
        dataset= dataset.drop(columns=cols)
    else:
        dataset= dataset.drop(columns=cols)
    return dataset

# find columns with too many Nan values
def Nan_values(dataset):
    column_names = dataset.columns.tolist()
    drop_columns = []
    for name in column_names:
        nan_count = dataset[name].isna().sum()
        print(f"column {name}: {nan_count}")
        if (nan_count/28000) > 0.5:
            print(f"Number of NaN values in column '{name}': {nan_count}")
            drop_columns.append(name)
    return drop_columns

def Replace_cate_to_value(column_name, dataset):
    # Extract unique category names from the column
    unique_categories = dataset[column_name].unique()

    # convert 'numpy.ndarray' in to a python list
    l = unique_categories.tolist()
    
    if 'MISSING' in l:
        l.remove('MISSING')
        l.sort(reverse=True)

    dic = { l[i]:i+1 for i in range(0, len(l))}

    # Replace values in the column based on the dictionary mapping
    dataset[column_name] = dataset[column_name].replace(dic)
    return dic, dataset


def Category_values(dataset):
    column_names = ['industry_sector', 'region', 'geo_area','external_score_ver03', 'province','juridical_form']
    dic = {}
    for column_name in column_names:
        category_dic, dataset = Replace_cate_to_value(column_name, dataset)
        dic[column_name] = category_dic
    return dic, dataset

def Replace_bool_toNumbers(dataset):
    dataset['cr_available'] = [int(dataset['cr_available'][i]) for i in range(len(dataset['cr_available']))]
    dataset['cr_available']
    return dataset

def mean_var03(dataset):
    s0, s1, c0, c1 = 0,0,0,0
    # unique_labels = dataset['target'].unique()
    for index, row in dataset.iterrows():
        if row['external_score_ver03'] != 'MISSING':
            if row['target'] == 0:
                s0 += row['external_score_ver03']
                c0 +=1
            elif row['target'] == 1:
                s1 +=  row['external_score_ver03']
                c1 += 1

    m0 = round(s0/c0)
    m1 = round(s1/c1)
    print(m0)
    print(m1)
    return m0,m1

def replace_missing_values(dataset, column_name ):
    s0, s1, c0, c1 = 0,0,0,0
    # unique_labels = dataset['target'].unique()
    for index, row in dataset.iterrows():
        if row[column_name] == 'MISSING':
            if row[column_name] == 0:
                s0 += row['external_score_ver03']
                c0 +=1
            elif row['target'] == 1:
                s1 +=  row['external_score_ver03']
                c1 += 1

    m0 = round(s0/c0)
    m1 = round(s1/c1)
    print(m0)
    print(m1)
    return m0,m1

def Replace_missing(dataset, m0, m1):
    # Assuming df is your DataFrame and 'column_to_change' is the column you want to change
    # 'condition_column' is the column based on which you want to change the content
    dataset.loc[(dataset['target'] == 1) & (dataset['external_score_ver03'] == 'MISSING'), 'external_score_ver03'] = m1
    dataset.loc[(dataset['target'] == 0) & (dataset['external_score_ver03'] == 'MISSING'), 'external_score_ver03'] = m0
    dataset['external_score_ver03']

    # For example, if you want to change the content of 'column_to_change' to 'new_value' where 'condition_column' is True
    # Replace 'new_value', 'column_to_change', and 'condition_column' with your actual values
    return dataset

# Drop columns 
train_dataset = Drop_unneed_columns(False,train_dataset)
drop_columns = Nan_values(train_dataset)
train_dataset = train_dataset.drop(columns=drop_columns)

# replace bool values to numerical ones
category_dics, train_dataset = Category_values(train_dataset)
train_dataset = Replace_bool_toNumbers(train_dataset)

# v03 column with missing values 
m0, m1= mean_var03(train_dataset)
train_dataset = Replace_missing(train_dataset, m0, m1)

# N O R M A L I Z E

def normalized_data(dataset):
    # Replace commas with periods in all columns
    dataset = dataset.replace(',', '.', regex=True)
    dataset = dataset.astype('float32')

    # check if the dataset has any nan value
    has_nan_values = dataset.isna().any().any()

    if has_nan_values:
        print("DataFrame contains NaN values.")
    else:
        print("DataFrame does not contain any NaN values.")

    return dataset

train_dataset = normalized_data(train_dataset)

# Define the neural network architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 8)  
        self.fc2 = nn.Linear(8,4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(4, 1)  # Output layer with 1 neuron for binary classification

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return torch.sigmoid(x)


accuracy_values = []
loss_values = []
X = train_dataset.iloc[:, :-1].to_numpy()
y = train_dataset.iloc[:, -1].to_numpy()

num_folds = 5
input_size = 39
num_epochs = 20
num_models = 5

kf = KFold(n_splits=num_folds, shuffle=True)

criterion = nn.BCELoss() 
l1_lambda = 0.001
l2_lambda = 0.001
fold_params = []

for model_index in range(num_models):

    for fold, (train_indices, val_indices) in enumerate(kf.split(X)):
        print(f'Fold {fold+1}/{num_folds}')

        # Split the data into training and validation sets
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]

        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
        
        model = NeuralNetwork(input_size)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # Train the neural network

        for epoch in range(num_epochs):
            # Forward pass
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor.view(-1, 1))
            loss_values.append(loss.item())

            l1_reg = torch.tensor(0., requires_grad=True)
            for param in model.parameters():
                l1_reg = l1_reg + torch.norm(param, p=1)
            loss = loss + l1_lambda * l1_reg

            # L2 regularization
            l2_reg = torch.tensor(0., requires_grad=True)
            for param in model.parameters():
                l2_reg = l2_reg + torch.norm(param, p=2)
            loss = loss + l2_lambda * l2_reg
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        fold_params.append(model.state_dict())
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Fold:{fold}')

        # Evaluate the model
        with torch.no_grad():
            # Predict probabilities on the test set
            outputs = model(X_val_tensor)
            predicted = (outputs >= 0.5).float()
            
            # Calculate accuracy
            accuracy = (predicted == y_val_tensor.view(-1, 1)).float().mean()
            accuracy_values.append(accuracy)
            print(f'Accuracy on test set: {accuracy.item()*100:.2f}%')
    torch.save(model.state_dict(), f'model_{model_index}.pth')

avg_params = {}

for key in fold_params[0].keys():
    avg_params[key] = torch.stack([params[key] for params in fold_params]).mean(dim=0)

# Create a new model with the average parameters
average_model = NeuralNetwork(input_size)
average_model.load_state_dict(avg_params)
print(f'Averagea ccuracy on test set: {np.array(accuracy_values).mean()*100:.2f}%')

