from .transformer import LTMModel
from .transformer_v2 import LTMModel_v2
from .transformer_no_attention import TransformerNoAttention
from .transformer_no_mask import TransformerNoMask
from .ann import ANNModel
from .cnn import CNNModel
from .lstm import LSTMModel


def get_model(name, num_features, config):

    if name == "transformer_v2":
        return LTMModel_v2(num_features, config)
    
    elif name == "transformer":
        return LTMModel(num_features, config)

    elif name == "transformer_no_attention":
        return TransformerNoAttention(num_features, config)

    elif name == "transformer_no_mask":
        return TransformerNoMask(num_features, config)

    elif name == "ann":
        return ANNModel(num_features)

    elif name == "cnn":
        return CNNModel(num_features)

    elif name == "lstm":
        return LSTMModel(num_features)

    else:
        raise ValueError(f"Unknown model: {name}")