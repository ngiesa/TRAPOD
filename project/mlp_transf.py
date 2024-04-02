import torch.nn as nn 
import torch
from torch import nn, Tensor
from model.mlp_model import MLP
import torch.nn.functional as F
import math
from model.transf_model import TimeSeriesTransformer
from model.mlp_model import MLP

class CombinedMLPTransformer(nn.Module):
    def __init__(self,
        transf_input_size: int,
        mlp_input_size: int,

        transf_batch_first: bool = True,
        transf_dim_val: int=32,  
        transf_n_encoder_layers: int=2,
        transf_n_heads: int=1,
        transf_dropout_encoder: float=0.2, 
        transf_dropout_pos_enc: float=0.1,
        transf_dim_feedforward_encoder: int=32,
        transf_max_seq_len: int=12,
        transf_num_predicted_features: int=1,
        transf_n_mlp_layers: int = 4,
        transf_n_mlp_nodes: int = 8,
        mlp_n_nodes=[4, 4, 4, 4],
        mlp_activation="sig",
        
        number_models: int = 2,
        number_outputs: int = 1
        ):

        super().__init__()

        self.transformer = TimeSeriesTransformer(
                                transf_input_size,
                                transf_batch_first,
                                transf_dim_val,  
                                transf_n_encoder_layers,
                                transf_n_heads,
                                transf_dropout_encoder, 
                                transf_dropout_pos_enc,
                                transf_dim_feedforward_encoder,
                                transf_max_seq_len,
                                transf_num_predicted_features,
                                transf_n_mlp_layers,
                                transf_n_mlp_nodes)

        self.mlp = MLP(mlp_input_size, mlp_n_nodes, mlp_activation)

        self.linear_layer = nn.Linear(
            in_features=number_models, 
            out_features=number_outputs
            )


    # defining forward using Transformers for sequences and MLP for static variables
    def forward(self, input_sequence: Tensor, input_static: Tensor, ) -> Tensor: # input seq 64, 6, 139, inp static = 64, 1, 339

        X_transf = self.transformer.forward(input=input_sequence)
        X_mlp = self.mlp.propagate(input_static)

        # squeece mlp output
        X_mlp = X_mlp.squeeze(1)

        # concat results form these two models
        X_cat = torch.cat((X_mlp, X_transf), 1)

        # linear layer 
        X = self.linear_layer(X_cat)

        # return logits from transf as well as mlp layer
        return X


