from .config import Config

config = Config()
config.exp_name = 'celeba'
config.shadow_dataset = 'celeba'
config.model_name = 'mbf_large'

config.target_ckpt_path = 'weights/backbone/mbf_large.pt'
config.shadow_ckpt_path = 'weights/shadow/mbf_large.ckpt'
