import torch.nn as nn 
import torch
from torch import nn, Tensor
from model.mlp_model import MLP
import torch.nn.functional as F
import math
from model.lstm_model import LSTM
from model.mlp_model import MLP

class CombinedMLPLSTM(nn.Module):
    def __init__(self,
        lstm_input_size: int,
        mlp_input_size: int,

        lstm_n_hidden: int = 6,
        lstm_n_layer: int= 2,
        lstm_drop_out: int=0.1,

        mlp_n_nodes=[4, 4, 4, 4],
        mlp_activation="sig",
        
        number_models: int = 2,
        number_outputs: int = 1
        ):

        super().__init__()

        self.lstm = LSTM(lstm_input_size, lstm_n_hidden, lstm_n_layer, lstm_drop_out)

        self.mlp = MLP(mlp_input_size, mlp_n_nodes, mlp_activation)

        self.linear_layer = nn.Linear(
            in_features=number_models, 
            out_features=number_outputs
            )


    # defining forward using lstm for sequences and MLP for static variables
    def forward(self, input_sequence: Tensor, input_static: Tensor, ) -> Tensor:

        X_lstm = self.lstm.forward(input_sequence)
        X_mlp = self.mlp.propagate(input_static)

        # squeece mlp output
        X_mlp = X_mlp.squeeze(1)

        # concat results form these two models
        X_cat = torch.cat((X_mlp, X_lstm), 1)

        # linear layer 
        X = self.linear_layer(X_cat)

        # return logits from transf as well as mlp layer
        return X


