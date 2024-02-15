from easydict import EasyDict as edict
config = edict()

# Data
config.train_ratio=0.8
config.input_size=112
config.num_classes=5749

# Model
config.model_name='mbf_large'
config.fp16=False
config.num_features=512
config.pretrain=True

# Training
config.batch_size=128
config.num_workers=16
config.lr=1e-3
config.max_epochs=100