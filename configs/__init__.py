from .config import Config

def get_config(config_name: str) -> Config:
    
    if config_name == 'default':
        return Config()

    elif config_name == 'mbf_large':
        from .mbf_large import config
        return config

    elif config_name == 'dp_mbf_large':
        from .dp_mbf_large import config
        return config
    
    else:
        raise ValueError(f'Unknown config name: {config_name}')

