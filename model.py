# ============================================== IMPORTS ============================================== #

from lightning.pytorch import LightningModule
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanSquaredError
from torchvision.models import resnet18
import torch
import torch.nn as nn
import torch.optim as optim

# ============================================== CLASSES ============================================== #

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.model = resnet18(weights='DEFAULT')
        num_features = self.model.fc.in_features # input to final layer
        self.model.fc = nn.Linear(in_features=num_features, out_features=1)

    def forward(self, images):
        return self.model(images)

class ResNetSentinel2(nn.Module):
    def __init__(self):
        super(ResNetSentinel2, self).__init__()
        self.model = resnet18(weights='DEFAULT')
        num_bands = 12
        original_weights = self.model.conv1.weight.clone()
        self.model.conv1 = nn.Conv2d(in_channels=num_bands,
                                     out_channels=self.model.conv1.out_channels,
                                     kernel_size=self.model.conv1.kernel_size,
                                     stride=self.model.conv1.stride,
                                     padding=self.model.conv1.padding,
                                     bias=self.model.conv1.bias)

        with torch.no_grad():
            self.model.conv1.weight[:, 3] = original_weights[:, 0]
            self.model.conv1.weight[:, 2] = original_weights[:, 1]
            self.model.conv1.weight[:, 1] = original_weights[:, 2]
            mean_original_weights = torch.mean(original_weights, dim=1)
            self.model.conv1.weight[:, 0], self.model.conv1.weight[:, 4:] = mean_original_weights, mean_original_weights
            self.model.conv1.weight *= 3/num_bands

        num_features = self.model.fc.in_features # input to final layer
        self.model.fc = nn.Linear(in_features=num_features, out_features=1)

    def forward(self, images):
        return self.model(images)

class Model(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.save_hyperparameters()
        self.configure_models()
        self.configure_metrics()

        self.criterion = nn.MSELoss()

    def configure_models(self):
        if self.hparams.model == 'resnet_rgb':
            self.model = ResNet()
        elif self.hparams.model == 'resnet_sentinel2':
            self.model = ResNetSentinel2()

    def configure_metrics(self):
        metrics = MetricCollection({'RMSE': MeanSquaredError(squared=False)})

        self.train_metrics = metrics.clone(prefix='Train ')
        self.val_metrics = metrics.clone(prefix='Val ')
        self.test_metrics = metrics.clone(prefix='Test ')

    def configure_optimizers(self):
        return optim.SGD(self.model.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        images, target = batch
        prediction = self(images, target)
        batch_size = images.shape[0]

        loss = self.criterion(prediction, target)
        self.log('Train loss', loss, batch_size=batch_size)
        self.train_metrics(prediction, target)
        self.log_dict(self.train_metrics, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        images, target = batch
        prediction = self(images, target)
        batch_size = images.shape[0]

        loss = self.criterion(prediction, target)
        self.log('Val loss', loss, batch_size=batch_size)
        self.val_metrics(prediction, target)
        self.log_dict(self.val_metrics, batch_size=batch_size)
