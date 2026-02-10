import torch
import torch.nn as nn

class OutfitLSTM(nn.Module):
    """
    BiLSTM that takes a sequence of item features (from ResNet)
    and outputs a single compatibility score for the outfit.
    """

    def __init__(self, input_dim=512, hidden_dim=128, num_layers=1, bidirectional=True):
        super(OutfitLSTM, self).__init__()
        
        # LSTM processes sequence of item features
        self.lstm = nn.LSTM(
            input_dim,            # 512-d features from ResNet
            hidden_dim,           # internal memory size
            num_layers,           # number of stacked LSTM layers
            batch_first=True,     # input shape: [batch, seq_len, feature_dim]
            bidirectional=bidirectional
        )

        # If bidirectional, output size doubles
        self.out_dim = hidden_dim * (2 if bidirectional else 1)

        # Dropout helps prevent overfitting
        self.dropout = nn.Dropout(p=0.3)

        # Final classifier: output a single score
        self.fc = nn.Linear(self.out_dim, 1)

    def forward(self, x):
        """
        Args:
            x: Tensor shape [batch, sequence_len, 512]
               Each sequence is an outfit; each timestep is an item
        
        Returns:
            score: Tensor shape [batch, 1]
        """
        # LSTM output for every item in the sequence
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_dim * directions]

        # Use the LAST output as outfit summary
        final_repr = lstm_out[:, -1, :]  # [batch, out_dim]

        # Regularization
        dropped = self.dropout(final_repr)

        # Map to single compatibility logit
        score = self.fc(dropped)  # [batch, 1]
        return score