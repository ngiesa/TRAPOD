
import torch.nn as nn 
import torch
from torch import nn, Tensor
from model.mlp_model import MLP
import torch.nn.functional as F
import math

class TimeSeriesTransformer(nn.Module):

    def __init__(self,
        input_size: int,
        batch_first: bool = True,
        dim_val: int= 64, # 64,
        n_encoder_layers: int=1,
        n_heads: int=1,
        dropout_encoder: float= 0.1,
        dropout_pos_enc: float=0.1,
        dim_feedforward_encoder: int= 64,
        max_seq_len: int=12,
        num_predicted_features: int=1,
        n_mlp_layers: int = 8,
        n_mlp_nodes: int = 4
        ):

        """
        Args:
            input_size: int, number of input features. 1 if univariate.
            dec_seq_len: int, the length of the input sequence fed to the decoder
            batch_first: if true the input format of X is batch_size x seq_len x input_variables
            dim_val: int, aka d_model. All sub-layers in the model produce 
                     outputs of dimension dim_val, hyperparameter 
            n_encoder_layers: int, number of stacked encoder layers in the encoder
            n_heads: int, the number of attention heads (aka parallel attention layers)
            dropout_encoder: float, the dropout rate of the encoder
            dropout_pos_enc: float, the dropout rate of the positional encoder
            max_seq_len: max sequence length of inputs
            dim_feedforward_encoder: int, number of neurons in the linear layer 
                                     of the encoder
            num_predicted_features: int, the number of features you want to predict.

        """

        super().__init__()

        # Creating the three linear layers for decoder, encoder and output 
        self.encoder_input_layer = nn.Linear(
            in_features=input_size, 
            out_features=dim_val)
        

        self.linear_mapping = nn.Linear(
            in_features=dim_val, 
            out_features=num_predicted_features
            )

        # Create positional encoder
        self.positional_encoding_layer = PositionalEncoder(
            d_model=dim_val,
            dropout=dropout_pos_enc
            )

        # Create encoder and decoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val, 
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            batch_first=batch_first
            )

        # Stack encoder and decoder layers in Transformer modules
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers, 
            #enable_nested_tensor=True,
            norm=None
            )

        # feed forward mlp applied to last layer learning spatial relationships
        self.mlp = MLP(input_size=max_seq_len, n_nodes=[n_mlp_nodes] * n_mlp_layers)
        #self.mlp = MLP(input_size=dim_val * max_seq_len, n_nodes=[n_mlp_nodes] * n_mlp_layers)


    def forward(self, input: Tensor, input_mask: Tensor=None) -> Tensor:
            """
            Returns logits as predictions per multivariable sequence
            
            Args:
                input: the encoder's  sequence.  if 
                    batch_first=True, (N, S, E) where S is the source sequence length, 
                    N is the batch size, and E is the number of features
                input_mask: the mask for the input sequence to handle sequences with different lengths

            """

            #print(torch.any(torch.isnan(input)))

            # input is batch first meaning batch, seq len, number of features
            X = self.encoder_input_layer(input)

            #print(torch.any(torch.isnan(X)))

            # adding positional encoding the dimentions are kept the same because the 
            # positional encoding is summed up with the acutal "embedding"
            X = self.positional_encoding_layer(X)

            # multihead attention only learns temporal relationships of sequences
            X = self.encoder(src=X)

            # print(torch.any(torch.isnan(X)))

            # replace nan in case there is too little data to learn dependence from 
            X = torch.nan_to_num(X)

            # feed encoder output to a linear mapping reducing hidden dimension to dim of prediction target
            X = self.linear_mapping(X)

            #print(torch.any(torch.isnan(X)))

            squeeze = torch.squeeze(X, 2)
            #flatten = torch.flatten(X, start_dim=1)

            # flatten to 20 * 128 = 2560
            # apply an MLP to aggregated hidden vectors squeezed in 2 dimensions batch dim and seq len * features
            X = self.mlp.propagate(squeeze) 
            
            # return logits 
            return X

class PositionalEncoder(nn.Module):

    def __init__(
        self, 
        dropout: float=0.1, 
        max_seq_len: int=100, #TODO maybe this is not set
        d_model: int=512,
        batch_first: bool=False
        ):

        """
        Parameters:
            dropout: the dropout rate
            max_seq_len: the maximum length of the input sequences
            d_model: The dimension of the output of sub-layers in the model 
                     (Vaswani et al, 2017)
        """

        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        # dimension adaption if batch_first then(batch_size, seq_len, hidden_size).
        self.x_dim = 0
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, 1, d_model)
        # positional encoding for even numbers
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        # positional encoding for uneven numbers
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val] or 
               [enc_seq_len, batch_size, dim_val]
        """

        print(x.size()) # pe tensor 5000, 1 , 128

        x = x + self.pe[:x.size(self.x_dim)]

        return self.dropout(x)