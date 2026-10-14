import torch
import torch.nn as nn

class SimpleRNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_size=256, dropout=0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=1,
            nonlinearity='tanh',
            batch_first=True
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x):
        # x: (batch_size, seq_len)
        embed = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        out, _ = self.rnn(embed)   # (batch_size, seq_len, hidden_size)
        last_timestep = out[:, -1, :]  # (batch_size, hidden_size)
        dropped = self.dropout(last_timestep)
        logits = self.fc(dropped)  # (batch_size, vocab_size)
        return logits

class BiRNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_size=256, dropout=0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=1,
            nonlinearity='tanh',
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(p=dropout)
        # Bidirectional concatenates forward and backward hidden states
        self.fc = nn.Linear(2 * hidden_size, vocab_size)
        
    def forward(self, x):
        # x: (batch_size, seq_len)
        embed = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        out, _ = self.rnn(embed)   # (batch_size, seq_len, 2 * hidden_size)
        last_timestep = out[:, -1, :]  # (batch_size, 2 * hidden_size)
        dropped = self.dropout(last_timestep)
        logits = self.fc(dropped)  # (batch_size, vocab_size)
        return logits

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_size=256, dropout=0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x):
        # x: (batch_size, seq_len)
        embed = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        out, _ = self.lstm(embed)  # (batch_size, seq_len, hidden_size)
        last_timestep = out[:, -1, :]  # (batch_size, hidden_size)
        dropped = self.dropout(last_timestep)
        logits = self.fc(dropped)  # (batch_size, vocab_size)
        return logits

class GRUModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_size=256, dropout=0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x):
        # x: (batch_size, seq_len)
        embed = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        out, _ = self.gru(embed)   # (batch_size, seq_len, hidden_size)
        last_timestep = out[:, -1, :]  # (batch_size, hidden_size)
        dropped = self.dropout(last_timestep)
        logits = self.fc(dropped)  # (batch_size, vocab_size)
        return logits

def count_parameters(model):
    """
    Returns total parameters and parameter breakdown by component.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    breakdown = {}
    for name, param in model.named_parameters():
        prefix = name.split(".")[0]
        breakdown[prefix] = breakdown.get(prefix, 0) + param.numel()
    return {
        "total": total_params,
        "trainable": trainable_params,
        "breakdown": breakdown
    }

def get_model_by_name(name, vocab_size, embed_dim=128, hidden_size=256, dropout=0.3):
    name_lower = name.lower().replace(" ", "").replace("_", "").replace("-", "")
    if name_lower in ["simplernn", "rnn"]:
        return SimpleRNNModel(vocab_size, embed_dim, hidden_size, dropout)
    elif name_lower in ["birnn", "bidirectionalrnn"]:
        return BiRNNModel(vocab_size, embed_dim, hidden_size, dropout)
    elif name_lower in ["lstm"]:
        return LSTMModel(vocab_size, embed_dim, hidden_size, dropout)
    elif name_lower in ["gru"]:
        return GRUModel(vocab_size, embed_dim, hidden_size, dropout)
    else:
        raise ValueError(f"Unknown model name: {name}")
