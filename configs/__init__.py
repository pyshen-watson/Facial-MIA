from .config import Config

def get_config(config_name: str) -> Config:
    
    if config_name == 'default':
        return Config()

    elif config_name == 'lfw':
        from .lfw import config
        return config
    
    elif config_name == 'ms1mv3':
        from .ms1mv3 import config
        return config
    
    else:
        raise ValueError(f'Unknown config name: {config_name}')

