import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

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

    def forward(self, x, lengths=None):
        """
        Args:
            x: Tensor shape [batch, sequence_len, 512]
               Each sequence is an outfit; each timestep is an item
            lengths: Tensor/List shape [batch], real (unpadded) sequence lengths
        
        Returns:
            score: Tensor shape [batch, 1]
        """
        if lengths is None:
            lengths = torch.full(
                (x.size(0),),
                x.size(1),
                dtype=torch.long,
                device=x.device,
            )
        elif not torch.is_tensor(lengths):
            lengths = torch.tensor(lengths, dtype=torch.long, device=x.device)
        else:
            lengths = lengths.to(device=x.device, dtype=torch.long)

        # Pack padded sequences so the LSTM ignores padded timesteps.
        packed_x = pack_padded_sequence(
            x,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (h_n, _) = self.lstm(packed_x)

        if self.lstm.bidirectional:
            h_n = h_n.view(self.lstm.num_layers, 2, x.size(0), self.lstm.hidden_size)
            final_repr = torch.cat((h_n[-1, 0], h_n[-1, 1]), dim=1)
        else:
            final_repr = h_n[-1]

        # Regularization
        dropped = self.dropout(final_repr)

        # Map to single compatibility logit
        score = self.fc(dropped)  # [batch, 1]
        return score
