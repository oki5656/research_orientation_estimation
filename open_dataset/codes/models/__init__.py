from models.lstm import get_lstm
from models.transformer_enc import TransAm
from models.transformer_encdec import Transformer
from models.IMUTransformerEncoder import IMUTransformerEncoder

MODEL_DICT = {
    "lstm": get_lstm,
    "transformer_enc": TransAm,
    "transformer_encdec": Transformer,
    "imu_transformer" : IMUTransformerEncoder,
}

def choose_model(model_name: str, train_columns_num: int, hidden_size: int, num_layers: int , nhead: int, output_dim: int, sequence_length: int, input_shift: int):
    if model_name not in MODEL_DICT:
        raise KeyError(f'Choose model_name from {MODEL_DICT.keys()}')

    if model_name == "lstm":
        return MODEL_DICT[model_name](train_columns_num, hidden_size, num_layers, output_dim, sequence_length)
    elif model_name == "transformer_enc":
        return MODEL_DICT[model_name](feature_size=6)
    elif model_name == "transformer_encdec":
        # return MODEL_DICT[model_name](d_model=6, nhead=3, seq_len=sequence_length, input_shift=input_shift)
        return MODEL_DICT[model_name](d_model=6, nhead=nhead, seq_len=sequence_length, input_shift=input_shift)
    elif model_name == "imu_transformer":
        return MODEL_DICT[model_name](seq_length=sequence_length)