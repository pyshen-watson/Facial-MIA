from typing import Dict
import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from .mobilefacenet import get_mbf, get_mbf_large, MobileFaceNet


def get_backbone(model_name='mbf', fp16=False, num_features=512, pretrain=True) -> MobileFaceNet:
    """
    This function returns a backbone model based on the given model_name. It has parameters model_name, fp16, num_features, and pretrain. The default values for fp16 and pretrain are False and True, respectively. The default value for num_features is 512. It returns the backbone model.
    """
    
    if model_name == 'mbf':
        backbone = get_mbf(fp16, num_features=num_features)
        ckpt_path =  'weights/mfn.pt'

    elif model_name == 'mbf_large':
        backbone = get_mbf_large(fp16, num_features=num_features)
        ckpt_path =  'weights/mfn_large.pt'
    
    else:
        raise ValueError(f'Unknown model name: {model_name}')

    if pretrain:
        backbone.load_state_dict(torch.load(ckpt_path))

    return backbone


class FaceRecognitionModule(pl.LightningModule):
    def __init__(self, backbone:nn.Module, config: Dict):

        super().__init__()
        self.save_hyperparameters(config)
        self.backbone = backbone
        self.classifier = nn.Linear(config.num_features, config.num_classes)
        
        self.criterion = nn.CrossEntropyLoss()
        self.acc = torchmetrics.Accuracy(task='multiclass', num_classes=config.num_classes)
        self.lr = config.lr

        # Freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Pass inputs through the backbone and the classifier
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def calc_loss(self, x, y):
        
        # Prepare data
        logits = self(x)
        preds = torch.softmax(logits, dim=1)
        labels = y
        
        # Calculate loss and metrics
        loss = self.criterion(preds, labels)
        acc = self.acc(preds, labels)

        return {
            'loss': loss,
            'acc': acc
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        results = self.calc_loss(x, y)
        self.log('train_loss', results['loss'], prog_bar=True)
        self.log('train_acc', results['acc'], prog_bar=True)
        return results
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        results = self.calc_loss(x, y)
        self.log('val_loss', results['loss'], prog_bar=True)
        self.log('val_acc', results['acc'], prog_bar=True)
        return results

    def configure_optimizers(self):
        # Optimize only the classifier parameters
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.lr)
        return optimizer