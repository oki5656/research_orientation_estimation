from models.lstm import get_lstm
from models.transformer_enc import TransAm
from models.transformer_encdec import Transformer

MODEL_DICT = {
    "lstm": get_lstm,
    "transformer_enc": TransAm,
    "transformer_encdec": Transformer,
}

def choose_model(model_name: str, train_columns_num: int, hidden_size: int, num_layers: int , output_dim: int, sequence_length: int):
    if model_name not in MODEL_DICT:
        raise KeyError(f'Choose model_name from {MODEL_DICT.keys()}')

    if model_name == "lstm":
        return MODEL_DICT[model_name](train_columns_num, hidden_size, num_layers, output_dim, sequence_length)
    elif model_name == "transformer_enc":
        return MODEL_DICT[model_name](feature_size=6)
    elif model_name == "transformer_encdec":
        return MODEL_DICT[model_name](d_model = 6, nhead=3)