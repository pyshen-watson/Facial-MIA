from easydict import EasyDict as edict
config = edict()
config.exp_name='Untitled-Experiment'

# Data
config.train_val_ratio=0.5
config.input_size=112 
config.num_classes=5749 # lfw

# Model
config.model_name='mbf_large'
config.fp16=False
config.num_features=512
config.pretrain=True

# Training
config.batch_size=128
config.num_workers=16
config.lr=1e-3
config.max_epochs=50