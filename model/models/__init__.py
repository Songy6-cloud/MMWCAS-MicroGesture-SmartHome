from .lptf_net import LPTFNet

def build_model(model_type='lptf_net', **config):
    if model_type != 'lptf_net':
        raise ValueError(model_type)
    return LPTFNet(**config)
