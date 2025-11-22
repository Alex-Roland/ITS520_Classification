import numpy as np
import torch
import pandas as pd
import sklearn
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from mlxtend.plotting import heatmap
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import xgboost as xgb
import onnxruntime as rt
import onnxmltools
from skl2onnx.common.data_types import FloatTensorType

# Parameters
batch_size = 32
learning_rate = 0.0003
N_epochs = 1000
epsilon = 0.0001

# Read Data
path_data = 'all_data_merged.csv'
temp_raw_data_df = pd.read_csv(path_data, delimiter=",")

temp_raw_data_df.season = temp_raw_data_df.season.map({'winter':0, 'spring':1, 'summer':2, 'fall':3})

headers_list = temp_raw_data_df.columns.values.tolist()

# Data Analysis
cm = np.corrcoef(temp_raw_data_df[headers_list].values.T)
hm = heatmap(cm, row_names=headers_list, column_names=headers_list, figsize=(15,10))
plt.show()

# Process Data
temp_raw_data_np = temp_raw_data_df.to_numpy()

X = temp_raw_data_np[:, :-1]
y = temp_raw_data_np[:, 4:5]

y = y.astype(int)

the_set = np.unique(y)

random_seed = int(random.random() * 100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

# Fix in case float64 error
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)

X_train_tr = torch.from_numpy(X_train)
X_test_tr = torch.from_numpy(X_test)
y_train_tr = torch.from_numpy(y_train)
y_test_tr = torch.from_numpy(y_test)

# Normalization
x_means = X_train_tr.mean(0, keepdim=True)
x_deviations = X_train_tr.std(0, keepdim=True) + epsilon

# Create the DataLoader
temp_train_list = [(X_train_tr[i], y_train_tr[i].item()) for i in range(X_train.shape[0])]
temp_test_list = [(X_test_tr[i], y_test_tr[i].item()) for i in range(X_test.shape[0])]
train_dl = torch.utils.data.DataLoader(temp_train_list, batch_size=batch_size, shuffle=True)
all_test_data = X_test.shape[0]
test_dl = torch.utils.data.DataLoader(temp_test_list, batch_size=all_test_data, shuffle=True)

# Neural Network Architectures

## MLP
class MLP_Net(nn.Module):
    ## init the class
    def __init__(self, x_means, x_deviations):
        super().__init__()
        
        self.x_means      = x_means
        self.x_deviations = x_deviations
        
        self.linear1 = nn.Linear(4, 4)
        self.act1    = nn.Sigmoid()
        self.linear2 = nn.Linear(4, 4)
        self.act2    = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.25)
        
    ## perform inference
    def forward(self, x):
        x = (x - self.x_means) / self.x_deviations
        
        x = self.linear1(x)
        x = self.act1(x)
        ## x = self.dropout(x)
        x = self.linear2(x)

        y_pred = self.act2(x)
        return y_pred

## Deep Learning with hidden layers
class DL_Net(nn.Module):
    ## init the class
    def __init__(self, x_means, x_deviations):
        super().__init__()
        
        self.x_means      = x_means
        self.x_deviations = x_deviations
        
        self.linear1 = nn.Linear(4, 16)
        self.act1    = nn.ReLU()
        self.linear2 = nn.Linear(16, 8)
        self.act2    = nn.ReLU()
        self.linear3 = nn.Linear(8, 4)
        self.act3    = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.25)
        
    ## perform inference
    def forward(self, x):
        x = (x - self.x_means) / self.x_deviations
        
        x = self.linear1(x)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.act2(x)
        x = self.dropout(x)
        x = self.linear3(x)

        y_pred = self.act3(x)
        return y_pred

# Training Loop
def training_loop(N_Epochs, model, loss_fn, opt):
    for epoch in range(N_Epochs):
        for xb, yb in train_dl:
            y_pred = model(xb)
            loss   = loss_fn(y_pred, yb)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
        if epoch % 20 == 0:
            print(epoch, "loss=", loss)

# model = MLP_Net(x_means, x_deviations)
model = DL_Net(x_means, x_deviations)

opt     = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

training_loop(N_epochs, model, loss_fn, opt)

# Evaluate Model
def print_metrics_function(y_test, y_pred):
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print("Confusion Matrix:")
    print(confmat)
    print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred, average='weighted'))
    print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred, average='weighted'))
    print('F1-measure: %.3f' % f1_score(y_true=y_test, y_pred=y_pred, average='weighted'))

with torch.no_grad():
    for x_real, y_real in test_dl:
        y_pred = model(x_real)
        vals, indeces = torch.max(y_pred, dim=1)
        preds = indeces
        print_metrics_function(y_real, preds)

# Deploy PyTorch Model
model.eval()

dummy_input = torch.randn(1, 4)

input_names  = ["input1"]
output_names = ["output1"]

torch.onnx.export(
        model,
        dummy_input,
        "temperature_humidity_data.onnx",
        input_names = input_names,
        output_names = output_names,
        opset_version=15,
        do_constant_folding=True,
        external_data=False,
        dynamic_axes={
            "input1": {0: "batch"},
            "output1": {0: "batch"}
        }
)
print("ONNX model saved")