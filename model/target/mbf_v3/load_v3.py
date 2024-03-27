from pathlib import Path
import torch
# v3.0.0
from backbone.mobilefacenet import get_mbf_large
from backbone.noise import NoisyActivation
from model.target.utils_v3 import images_to_batch
import time
abs_path = Path('./').absolute()
v300_mobilefacenet_ckpt_path = str(abs_path / 'ckpt' / 'model.pt')
v300_noise_ckpt_path = str(abs_path / 'ckpt' / 'noise.pt')

# noise
noise_model = NoisyActivation()
noise_model.load_state_dict(torch.load(str(v300_noise_ckpt_path), map_location='cuda:0'))
noise_model.cuda()
noise_model.eval()
# iresnet100
v300_MOBILE_LARGE = get_mbf_large(fp16=False, num_features=512)
v300_MOBILE_LARGE.load_state_dict(torch.load(str(v300_mobilefacenet_ckpt_path), map_location='cuda:0'))
v300_MOBILE_LARGE.cuda()
v300_MOBILE_LARGE.eval()

x = torch.randn((1, 3, 112, 112)).cuda()
for _ in range(100):
    torch.cuda.synchronize()
    tic = time.time()
    data = images_to_batch(x)
    out = noise_model(data)

    reg_out = v300_MOBILE_LARGE(out)
    torch.cuda.synchronize()
    print('forward time: {:.4f}s'.format(time.time() - tic), ", FPS:", (int(1 / (time.time() - tic))))
    print(reg_out.shape)