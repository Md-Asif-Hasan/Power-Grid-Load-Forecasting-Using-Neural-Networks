import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_size=128, num_layers=2, dropout=0.2, output_dim=24):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class RNNSeq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, horizon=24, cell="LSTM"):
        super().__init__()
        rnn_cls = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[cell]
        self.encoder = rnn_cls(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.decoder = rnn_cls(1, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.proj = nn.Linear(hidden_size, 1)
        self.horizon = horizon

    def forward(self, x):
        _, state = self.encoder(x)
        batch = x.size(0)
        dec_in = torch.zeros(batch, 1, 1, device=x.device)
        outputs = []
        for _ in range(self.horizon):
            dec_out, state = self.decoder(dec_in, state)
            y = self.proj(dec_out)
            outputs.append(y)
            dec_in = y
        return torch.cat(outputs, dim=1).squeeze(-1)

def build_model(model_type, input_size, hidden_size, num_layers, dropout, horizon):
    if model_type == "MLP":
        return MLP(input_dim=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, output_dim=horizon)
    elif model_type in ["RNN", "LSTM", "GRU"]:
        return RNNSeq2Seq(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, horizon=horizon, cell=model_type)
    else:
        raise ValueError(f"Unknown model_type={model_type}")
