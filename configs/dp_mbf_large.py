from .config import Config

config = Config()
config.exp_name = 'dp_mbf_large'

config.model_name = 'dp_mbf_large'
config.target_ckpt_path = 'weights/backbone/dp_mbf_large.pt'
config.shadow_ckpt_path = 'weights/shadow/dp_mbf_large.ckpt'
