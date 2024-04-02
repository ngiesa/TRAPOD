from unicodedata import bidirectional
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, n_hidden=6, n_layer=2, drop_out=0.1):
        super().__init__()

        # batch_first = True sets input format to (batch_size, seq_len, features) 
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=n_hidden,
            num_layers=n_layer,
            batch_first=True,
            dropout=drop_out,
            bidirectional=False,
        )

        # sigmoid focal is applied on logits from last linear layer in loss
        self.linear = nn.Linear(n_hidden, 1)

    def forward(self, x):

        _, (h, _) = self.lstm(x)
        lin_act = self.linear(h[-1])
        return lin_act
