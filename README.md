# Model Inversion Attack on facial recognition system under blackbox setting

## Installation
### Environment
- OS: Ubuntu 20.04 LTS
- Python: `3.9.19`

### Installation Step
1. Create a Conda environment: `conda create -n MIA python=3.9.19`
1. Activate the Conda environment: `conda activate MIA`
1. Install the dependencies: `pip install -r requirements.txt`

## Methodology
This project conducts the model inversion attack for face recognition system under the black-box setting. Here is the term we use:
- **Model Inversion Attack:** To figure out the private training data of a model. In this project, we implement a MIA aat inference time, which is harder than conventional MIA because all the information we have is the original input and the outpur feature map. We know nothing about the structure of the model or its weight.
- **Target model:** The victim model that we want to attack. In our case, a facial recognition model.
- **Attack model:** The model trained for inverse the output of target model to the original input. See [this paper](https://ieeexplore.ieee.org/abstract/document/9897535).

The whole pipeline looks like this:
1. There is an image contain some people, we will do detection and cropping their face, and resize to 112x112.
1. The face image turn to a tensor in the shape of **(N, 3, 112, 112)**, which N stands for the batch size.
1. We feed this batch to the target model, and receive the output in the shape of **(N, 512)**.
1. The output feature map will then be fed into the attack model, and output a reconstructed image in the same shape with the original input image.
1. In our project, we compare the original and reconstructed image by three measures:
    - $L_{MSE}=||x-x'||^2$
    - $L_{DDSIM}=\frac{1}{2}[1-SSIM(x,x')]$
    - $L_{id}=||f(x)-f(x')||^2$, where $f$ is the target model.


## Project Structure
- `dataset/` - Define the preprocessing, collecting function of the datasets
- `model/` - Define the structure of target and attack models
- `weights/` - Save the weights of target/attack models
- `export.onnx` - Export a target/attack weight ends with `.pt` to `.onnx` format.
- `mia_train.py` - Specify a target model and train a corresponding attack model.
- `mia_eval.py` - Evaluate a attack model only.

## Run

Execution: `python mia_train.py` / `python mia_eval.py` 

**Before the training process, we need to configure three things:**

### TargetModel
Example:
```python
target = TargetModel(TargetType.MBF_LARGE_V1).load('weights/target/mbf_large_v1.pt')
```
- `TargetModel`: An encapsulation of the target model, provide a universal interface for different architecture of models. The first parameter should be a member of `TargetType`, you should add a new enumeration in `model/target/target.py` if you want to use a customized target model.
### AttackModel
Example:
```python
attack = AttackModel(AttackType.IDIAP, dssim_weight=0.25, id_weight=0.25)
```
- `AttackModel`: An encapsulation of the attack model, provide a universal interface for different architecture of models. The first parameter should be a member of `AttackType`, you should add a new enumeration in `model/attack/attack.py` if you want to use a customized target model.
- `dssim_weight` and `id_weight` are the weights in the loss function.
- **NOTE:** Currently, we only support one attack model from [idiap](https://ieeexplore.ieee.org/abstract/document/9897535).

After setting the target and attack model, put them together into `ModelInversionAttackModule`
```python
mia_model = ModelInversionAttackModule(target, attack)
```
### FaceDataModule
```python
dm = FaceDataModule(root_dir, size, batch_size, train_ratio)
```
This module is an abstract of the dataset. Here is the parameters:
- `root_dir`: The path to a directory contain the images. The `root_dir` is in this structure:
    ```
    root_dir
        ├── 0
        |   ├── 00001.jpg
        |   └── 00002.jpg
        ├── 1
        |   ├── 00003.jpg
        |   ├── 00004.jpg
        |   └── 00005.jpg
        ...
    ```
    The images are divided by their labels. If the dataset does not have a label, you still need to put all the image inside a label named `0`.
- `size`: The input size of the target model. All the images will resize to this size.
- `batch_size`: Here is some recommanded batch size

| GPU VRAM | Batch Size|
| :---------------: | :--------:|
| 24 GiB| 160|
| 12 GiB| 80|
| 8 GiB | 50|
- `train_ratio`: a float number between 0~1 for the proportion of training dataset. The remain will be divided equally to validation set and test set. For example, if `train_ratio=0.8`, the dataset will be split to 80%, 10% and 10% for train, validation and test dataset.

## Deployment
To export the `.pt` files into `.onnx` format, just modify the pathes to target and attack models, then run `python export_onnx.py`