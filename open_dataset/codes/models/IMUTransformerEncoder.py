import json
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class IMUTransformerEncoder(nn.Module):

    def __init__(self, seq_length, input_dim=6, transformer_dim=64, encode_position=True, nhead=8, dim_feedforward=128, transformer_dropout=0.1,
                 transformer_activation="gelu", num_encoder_layers=6):
        """
        config: (dict) configuration of the model
        """
        super().__init__()

        self.transformer_dim = transformer_dim # config.get("transformer_dim")

        self.input_proj = nn.Sequential(#nn.Conv1d(config.get("input_dim"), self.transformer_dim, 1), nn.GELU(),
                                        nn.Conv1d(input_dim, self.transformer_dim, 1), nn.GELU(),
                                        nn.Conv1d(self.transformer_dim, self.transformer_dim, 1), nn.GELU(),
                                        nn.Conv1d(self.transformer_dim, self.transformer_dim, 1), nn.GELU(),
                                        nn.Conv1d(self.transformer_dim, self.transformer_dim, 1), nn.GELU())

        self.window_size = seq_length # config.get("window_size")
        self.encode_position = encode_position # config.get("encode_position")
        encoder_layer = TransformerEncoderLayer(d_model = self.transformer_dim,
                                       nhead = nhead, # config.get("nhead"),
                                       dim_feedforward = dim_feedforward, # config.get("dim_feedforward"),
                                       dropout = transformer_dropout, # config.get("transformer_dropout"),
                                       activation = transformer_activation # config.get("transformer_activation")
                                       )

        self.transformer_encoder = TransformerEncoder(encoder_layer,
                                              num_layers = num_encoder_layers, # config.get("num_encoder_layers"),
                                              norm = nn.LayerNorm(self.transformer_dim))
        self.cls_token = nn.Parameter(torch.zeros((1, self.transformer_dim)), requires_grad=True)

        if self.encode_position:
            self.position_embed = nn.Parameter(torch.randn(self.window_size + 1, 1, self.transformer_dim))

        # num_classes =  config.get("num_classes")
        out_feature = 3
        self.imu_head = nn.Sequential(
            nn.LayerNorm(self.transformer_dim),
            nn.Linear(self.transformer_dim,  self.transformer_dim//4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.transformer_dim//4,  out_feature)
        )
        # self.log_softmax = nn.LogSoftmax(dim=1)
        self.linear = nn.Linear(64, 3)

        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src):
        # src = data.get('imu')  # Shape N x S x C with S = sequence length, N = batch size, C = channels
        # print("forward src shape", src.shape)
        # print("src.transpose(1, 2).shape", src.transpose(1, 2).shape)

        # change IMUTransformer shape to my repositry
        src = src.permute(1, 0, 2)

        # Embed in a high dimensional space and reshape to Transformer's expected shape
        src = self.input_proj(src.transpose(1, 2)).permute(2, 0, 1)

        # Prepend class token
        cls_token = self.cls_token.unsqueeze(1).repeat(1, src.shape[1], 1)
        src = torch.cat([cls_token, src])

        # Add the position embedding
        if self.encode_position:
            src += self.position_embed

        # Transformer Encoder pass
        target = self.transformer_encoder(src)[0]

        # original shape adjustment
        # target = self.linear(target)

        # Class probability
        # target = self.log_softmax(self.imu_head(target))
        target = self.imu_head(target)

        return target

def get_activation(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "gelu":
        return nn.GELU()
    raise RuntimeError("Activation {} not supported".format(activation))
