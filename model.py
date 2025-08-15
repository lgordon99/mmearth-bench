# ============================================== IMPORTS ============================================== #

from collections import OrderedDict
from convnextv2 import ConvNeXtV2, load_custom_checkpoint
from lightning.pytorch import LightningModule
from timm.models.layers import trunc_normal_
from torchmetrics import Metric, MetricCollection, Recall
from torchmetrics.classification import MultilabelRecall, MultilabelAveragePrecision
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score
from torchvision.models import resnet50
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim
import utils
import wandb

# ============================================== GLOBAL VARIABLES ============================================== #

num_logged_images = 25
fontsize = 50
pad = 10
data_dir_path = os.environ['DATA_DIR_PATH']
no_data_values = utils.read_json(f'{data_dir_path}/no_data_values.json')
biomass_no_data_value = -9999
architecture_properties = {'ResNet50': {'num_image_channels': 3, 'embedding_dim': 2048}, 'DINOv2': {'num_image_channels': 3, 'embedding_dim': 384}, 'MPMAE': {'num_image_channels': 12, 'embedding_dim': 320}}

# ============================================== CLASSES ============================================== #

class MeanError(Metric):
    def __init__(self):
        super().__init__()

        self.add_state('error_sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        self.error_sum += torch.sum(preds - target)
        self.total += target.numel()

    def compute(self):
        return self.error_sum / self.total

class MPMAE(nn.Module):
    def __init__(self, num_classes, pixelwise, pretrained):
        super(MPMAE, self).__init__()

        self.pixelwise = pixelwise
        self.model = ConvNeXtV2(num_classes=num_classes)

        if pixelwise:
            self.model.head = nn.Identity() # removes the classifier layer
            self.model.forward_features = self._pixelwise_forward_features # replaces the forward_features method with a pixelwise version

        if pretrained:
            checkpoint_path = f'{data_dir_path}/all_mod_atto_1M_64_uncertainty_56-8.pth' # Vishal's checkpoint
            load_custom_checkpoint(self.model, checkpoint_path) # freezing and unfreezing is done in this function

    def _pixelwise_forward_features(self, x):
        if self.model.use_orig_stem:
            x = self.model.stem_orig(x)
        else:
            x = self.model.initial_conv(x)
            x = self.model.stem(x)

        x = self.model.stages[0](x)

        for i in range(3):
            x = self.model.downsample_layers[i](x)
            x = self.model.stages[i + 1](x)

        return self.model.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def forward(self, images):
        return self.model(images['Sentinel2'])

class DINOv2(nn.Module):
    def __init__(self, num_classes, pixelwise, pretrained):
        super(DINOv2, self).__init__()

        self.pixelwise = pixelwise
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        self.model.interpolate_pos_encoding = self._deterministic_interpolate_pos_encoding

        if pixelwise:
            self.model.head = nn.Identity()
        else:
            self.model.head = nn.Linear(in_features=384, out_features=num_classes)

    def _deterministic_interpolate_pos_encoding(self, x, w, h):
        '''Deterministic version using CPU for bicubic interpolation'''

        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.model.pos_embed.shape[1] - 1

        if npatch == N and w == h:
            return self.model.pos_embed

        pos_embed = self.model.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.model.patch_size
        h0 = h // self.model.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M

        kwargs = {}

        if self.model.interpolate_offset:
            sx = float(w0 + self.model.interpolate_offset) / M
            sy = float(h0 + self.model.interpolate_offset) / M
            kwargs['scale_factor'] = (sx, sy)
        else:
            kwargs['size'] = (w0, h0) # specifies an output size instead of a scale factor

        patch_pos_embed = nn.functional.interpolate(patch_pos_embed.cpu().reshape(1, M, M, dim).permute(0, 3, 1, 2),
                                                    mode='bicubic',
                                                    antialias=self.model.interpolate_antialias,
                                                    **kwargs)
        patch_pos_embed = patch_pos_embed.to(x.device)
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def forward(self, images):
        images = images['Sentinel2'][:, [3,2,1], :, :]
        images = torch.nn.functional.pad(images, (6, 6, 6, 6), mode='constant') # DinoV2 requires the image size to be divisible by 14

        if self.pixelwise:
            features = self.model.forward_features(images)['x_norm_patchtokens'].permute(0, 2, 1)

            return features.reshape(features.shape[0], features.shape[1], int(np.sqrt(features.shape[2])), int(np.sqrt(features.shape[2])))
        else:
            return self.model(images)

class TaskModalityEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        num_pixel_level_bands = 46
        num_tile_level_bands = 880

        self.encoder = smp.Unet(encoder_name='resnet50', encoder_weights='imagenet', in_channels=num_pixel_level_bands+num_tile_level_bands).encoder
        modality_names = list(no_data_values.keys())

        self.pixel_level_modality_names = modality_names[:6]
        self.tile_level_modality_names = modality_names[6:]

    def forward(self, task_modalities):
        pixel_level_modalities = torch.cat([task_modalities[modality] for modality in self.pixel_level_modality_names], dim=1)
        tile_level_modalities = torch.cat([task_modalities[modality] for modality in self.tile_level_modality_names], dim=1)
        tile_level_modalities_spatial = tile_level_modalities.view(*tile_level_modalities.shape, 1, 1).expand(*tile_level_modalities.shape[:2], *pixel_level_modalities.shape[2:]) # expands the tile-level modalities to match the shape of the pixel-level modalities
        modalities = torch.cat([pixel_level_modalities, tile_level_modalities_spatial], dim=1) # combines the pixel-level and tile-level modalities
        modality_embeddings = self.encoder(modalities)[-1]

        return modality_embeddings

class ResNet50Encoder(nn.Module):
    def __init__(self, pixelwise, pretrained):
        super().__init__()

        self.model = resnet50(weights='DEFAULT' if pretrained else None)
        self.model._forward_impl = self._forward_impl_encode # removes the avgpool, flatten, and fc layers

    def _forward_impl_encode(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        return x

    def forward(self, images):
        return self.model(images['Sentinel2']['data'][:, [3,2,1], :, :]) # extracts the RGB bands

class TaskDecoder(nn.Module):
    def __init__(self, architecture, pixelwise, adaptation_mode, num_classes):
        super().__init__()

        self.architecture = architecture
        self.num_classes = num_classes

        if architecture == 'ResNet50':
            self.decoder = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                         nn.Flatten(1),
                                         nn.LazyLinear(num_classes)) # infers in_features from input shape

    def forward(self, embeddings):
        task_prediction = self.decoder(embeddings)

        return task_prediction

class LinearDecoder(nn.Module):
    def __init__(self, image_size, num_image_channels, embedding_dim, out_channels):
        super().__init__()

        self.num_image_channels = num_image_channels
        self.upsample = nn.Upsample(size=image_size, mode='bilinear') # upsamples bilinearly to the image size
        self.convolution = nn.Conv2d(in_channels=num_image_channels+embedding_dim, out_channels=out_channels, kernel_size=1) # applies a linear layer to each pixel

    def forward(self, embeddings, images):
        images = images['Sentinel2']['data']

        if self.num_image_channels != 12:
            images = images[:, [3,2,1], :, :] # extracts the RGB bands

        upsampled_embeddings = self.upsample(embeddings) # upsamples the embeddings to the image size
        concatenated = torch.cat((images, upsampled_embeddings), dim=1) # concatenates along the channel dimension
        convolved = self.convolution(concatenated) # applies the convolution to the concatenated tensor

        return convolved

class TaskModalityDecoder(nn.Module):
    def __init__(self, num_image_channels, embedding_dim):
        super().__init__()

        self.task_modality_decoder = LinearDecoder(image_size=128, num_image_channels=num_image_channels, embedding_dim=embedding_dim, out_channels=922) # 922 is the total number of bands in the task modalities excluding the NaN bands in the categorical modalities
        self.modality_band_indices = {'Sentinel2': [0, 12],
                                      'Sentinel1': [12, 20],
                                      'AsterDEM': [20, 22],
                                      'ETH_GCH': [22, 24],
                                      'DynamicWorld': [24, 33],
                                      'ESA_WorldCover': [33, 44],
                                      'precipitation': [44, 47],
                                      'temperature': [47, 56],
                                      'geolocation': [56, 60],
                                      'month': [60, 62],
                                      'biome': [62, 76],
                                      'ecoregion': [76, 922]}
        self.tile_level_modality_names = ['precipitation', 'temperature', 'geolocation', 'month', 'biome', 'ecoregion']

    def forward(self, embeddings, images):
        modality_reconstructions = self.task_modality_decoder(embeddings, images)
        modality_reconstructions = {modality: modality_reconstructions[:, indices[0]:indices[1]] for modality, indices in self.modality_band_indices.items()}
        modality_reconstructions = {modality: reconstruction.mean(dim=(2,3)) if modality in self.tile_level_modality_names else reconstruction for modality, reconstruction in modality_reconstructions.items()} # collapses the spatial dimensions for the tile-level modalities

        return modality_reconstructions

class ModalityReconstructionLossCalculator(nn.Module):
    def __init__(self):
        super().__init__()

        self.mse_loss = nn.MSELoss()
        self.categorical_modalities = ['DynamicWorld', 'ESA_WorldCover', 'biome', 'ecoregion']
        self.cross_entropy_losses = {modality: nn.CrossEntropyLoss(ignore_index=no_data_values[modality]) for modality in self.categorical_modalities}

    def forward(self, modality_reconstructions, images):
        modality_reconstruction_losses = {}

        for modality, reconstruction in modality_reconstructions.items():
            target = images[modality]['data']

            if modality in self.categorical_modalities: # categorical modalities
                modality_reconstruction_losses[modality] = self.cross_entropy_losses[modality](reconstruction.cpu(), target.squeeze().long().cpu()).to(reconstruction.device)
            else: # continuous modalities
                if 'valid_mask' in images[modality].keys():
                    valid_mask = images[modality]['valid_mask']
                    reconstruction = reconstruction[valid_mask]
                    target = target[valid_mask]

                modality_reconstruction_losses[modality] = self.mse_loss(reconstruction, target)

        return modality_reconstruction_losses

class SurrogateLossNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.surrogate_loss_network = nn.Sequential(nn.Linear(12, 64),
                                                    nn.ReLU(),
                                                    nn.Linear(64, 64),
                                                    nn.ReLU(),
                                                    nn.Linear(64, 1),
                                                    nn.Softplus()) # ensures the surrogate loss is positive

    def forward(self, modality_reconstruction_losses):
        return self.surrogate_loss_network(torch.stack(list(modality_reconstruction_losses.values())))

class EncoderDecoder(nn.Module):
    def __init__(self, architecture, adaptation_mode, pixelwise, num_classes, pretrained=None):
        super().__init__()

        self.adaptation_mode = adaptation_mode
        self.pixelwise = pixelwise
        self.encoder = globals()[f'{architecture}Encoder'](pixelwise, pretrained)
        num_image_channels = architecture_properties[architecture]['num_image_channels']
        embedding_dim = architecture_properties[architecture]['embedding_dim']

        if adaptation_mode == 'multimodal':
            self.task_modality_encoder = TaskModalityEncoder()
        elif adaptation_mode == 'maml':
            self.task_modality_decoder = TaskModalityDecoder(num_image_channels, embedding_dim)
            self.modality_reconstruction_loss_calculator = ModalityReconstructionLossCalculator()
            self.surrogate_loss_network = SurrogateLossNetwork()

        self.task_decoder = TaskDecoder(architecture, pixelwise, adaptation_mode, num_classes)

    def forward(self, images):
        embeddings = self.encoder(images)

        if self.adaptation_mode == 'multimodal':
            modality_embeddings = self.task_modality_encoder(images)
            embeddings = torch.cat([embeddings, modality_embeddings], dim=1)
        elif self.adaptation_mode == 'maml':
            surrogate_loss = self.surrogate_loss_network(self.modality_reconstruction_loss_calculator(modality_reconstructions=self.task_modality_decoder(embeddings, images), images=images))

        task_prediction = self.task_decoder(embeddings)

        return task_prediction

# class LinearDecoder(nn.Module):
#     def __init__(self, num_input_channels, image_size, feature_dim):
#         super().__init__()

#         self.num_input_channels = num_input_channels
#         self.image_size = image_size
#         self.upsample = nn.Upsample(size=image_size, mode='bilinear') # upsamples bilinearly to the image size
#         self.convolution = nn.Conv2d(in_channels=num_input_channels+feature_dim, out_channels=1, kernel_size=1) # applies a linear layer to each pixel

#     def forward(self, images, features):
#         if self.num_input_channels != 12:
#             images = images[:, [3,2,1], :, :] # extracts the RGB bands

#         if self.image_size != 128:
#             images = torch.nn.functional.pad(images, (6, 6, 6, 6), mode='constant') # DinoV2 requires the image size to be divisible by 14

#         upsampled_features = self.upsample(features)
#         concatenated = torch.cat((images, upsampled_features), dim=1) # concatenates along the channel dimension
#         convolved = self.convolution(concatenated) # applies the convolution to the concatenated tensor

#         return convolved

# class EncoderDecoder(nn.Module):
#     def __init__(self, model_class_name, adaptation_mode, pixelwise, num_classes, pretrained=None):
#         super(EncoderDecoder, self).__init__()

#         self.pixelwise = pixelwise
#         self.model = globals()[model_class_name](num_classes, pixelwise, pretrained)

#         if pixelwise:
#             num_input_channels = 12 if model_class_name == 'MPMAE' else 3 # all Sentinel-2 bands or just RGB
#             image_size = 140 if model_class_name == 'DINOv2' else 128

#             if model_class_name == 'ResNet50':
#                 feature_dim = 2048
#             elif model_class_name == 'DINOv2':
#                 feature_dim = 384
#             elif model_class_name == 'MPMAE':
#                 feature_dim = 320

#             self.decoder = LinearDecoder(num_input_channels, image_size, feature_dim) # creates the decoder with the number of features

#     def forward(self, images):
#         if self.pixelwise:
#             return self.decoder(images=images['Sentinel2'], features=self.model(images)) # applies the decoder to the features
#         else:
#             return self.model(images)

class Model(LightningModule):
    def __init__(self, task, model, adaptation_mode, tuning_mode, decay_factor, max_lr, weight_decay, warmup_epochs, num_train_batches, min_lr, epochs):
        super().__init__()

        self.save_hyperparameters()
        self.configure_models()
        self.configure_metrics()

        if self.hparams.model == 'modalityreconstruction':
            self.criterion = ModalityReconstructionNetworkLoss()
        elif task == 'species': # multi-label classification
            self.criterion = nn.BCEWithLogitsLoss()
        else: # regression
            self.criterion = nn.MSELoss()

    def configure_models(self):
        pixelwise = True if self.hparams.task == 'biomass' else False
        num_classes = 100 if self.hparams.task == 'species' else 1

        if self.hparams.model == 'resnet50':
            self.model = EncoderDecoder(architecture='ResNet50', adaptation_mode=self.hparams.adaptation_mode, pixelwise=pixelwise, num_classes=num_classes, pretrained=False)
        elif self.hparams.model == 'resnet50_imagenet':
            self.model = EncoderDecoder(architecture='ResNet50', adaptation_mode=self.hparams.adaptation_mode, pixelwise=pixelwise, num_classes=num_classes, pretrained=True)
        elif self.hparams.model == 'dinov2':
            self.model = EncoderDecoder(architecture='DINOv2', adaptation_mode=self.hparams.adaptation_mode, pixelwise=pixelwise, num_classes=num_classes)
        elif self.hparams.model == 'mpmae':
            self.model = EncoderDecoder(architecture='MPMAE', adaptation_mode=self.hparams.adaptation_mode, pixelwise=pixelwise, num_classes=num_classes, pretrained=False)
        elif self.hparams.model == 'mpmae_mmearth':
            self.model = EncoderDecoder(architecture='MPMAE', adaptation_mode=self.hparams.adaptation_mode, pixelwise=pixelwise, num_classes=num_classes, pretrained=True)
        elif self.hparams.model == 'modalityreconstruction':
            self.model = ModalityReconstructionNetwork()

    def configure_metrics(self):
        if self.hparams.task == 'species':
            num_labels = 100
            metrics = MetricCollection({'Recall': MultilabelRecall(num_labels),
                                        'MAP': MultilabelAveragePrecision(num_labels)})
        else:
            metrics = MetricCollection({'RMSE': MeanSquaredError(squared=False),
                                        'MAE': MeanAbsoluteError(),
                                        'ME': MeanError(),
                                        'R2': R2Score()})

        for split in ['train', 'val', 'random_test', 'geographic_test']:
            setattr(self, f'{split}_metrics', metrics.clone(prefix=f'{split.replace("_", " ").capitalize()} '))

    def configure_parameters(self, max_lr):
        if self.hparams.tuning_mode == 'lp':
            # freeze all parameters
            for param in self.model.parameters():
                param.requires_grad = False

            # unfreeze the final or decoder layers
            if self.hparams.task == 'biomass':
                parameters_to_unfreeze = self.model.decoder.parameters()
            else:
                if 'resnet' in self.hparams.model:
                    parameters_to_unfreeze = self.model.model.model.fc.parameters() # final layer
                elif 'dino' in self.hparams.model or 'mpmae' in self.hparams.model:
                    parameters_to_unfreeze = self.model.model.model.head.parameters() # final layer

            for param in parameters_to_unfreeze:
                param.requires_grad = True

            if self.hparams.task == 'biomass':
                assert sum(p.numel() for p in self.model.decoder.parameters()) == sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        elif self.hparams.tuning_mode == 'llrd':
            layer_names = []

            for name, _ in self.model.model.model.named_parameters():
                parts = name.split('.')
                num_digits = sum(1 for char in name if char.isdigit())

                if 'resnet' in self.hparams.model:
                    if num_digits == 0 or num_digits == 1:
                        layer_name = parts[0]
                    elif num_digits == 3:
                        layer_name = f'{parts[0]}.{parts[1]}'
                elif 'mpmae' in self.hparams.model:
                    if num_digits == 0:
                        layer_name = parts[0]
                    elif num_digits == 1:
                        layer_name = f'{parts[0]}.{parts[1]}'
                    elif num_digits == 2:
                        layer_name = f'{parts[0]}.{parts[1]}.{parts[2]}'
                elif 'dino' in self.hparams.model:
                    if len(parts) == 1:
                        layer_name = 'embedding_tokens'
                    elif num_digits == 0:
                        layer_name = parts[0]
                    elif len(parts) == 4:
                        layer_name = f'{parts[0]}.{parts[1]}'

                if layer_name not in layer_names:
                    layer_names.append(layer_name)

            if self.hparams.task == 'biomass':
                layer_names.append('decoder.convolution')

            layer_names.reverse()
            print(f'{len(layer_names)} layer groups')
            print(layer_names)
            parameters = []

            for i, name in enumerate(layer_names):
                learning_rate = max_lr * self.hparams.decay_factor ** i
                print(f'{name}: {learning_rate}')

                if name == 'decoder.convolution':
                    parameters += [{'params': self.model.decoder.convolution.parameters(),
                                    'lr': learning_rate,
                                    'name': name}]
                else:
                    parameters += [{'params': [p for n, p in self.model.model.model.named_parameters() if n == name or (len(n.split(name)) > 1 and n.startswith(name) and n.split(name)[1][0] == '.') or (name == 'embedding_tokens' and (n == 'cls_token' or n == 'pos_embed' or n == 'register_tokens' or n == 'mask_token'))],
                                    'lr': learning_rate,
                                    'name': name}]

            assert sum(p.numel() for p in self.model.parameters()) == sum(p.numel() for group in parameters for p in group['params'])

            return parameters

        return self.model.parameters()

    def configure_optimizers(self):
        parameters = self.configure_parameters(max_lr=self.hparams.max_lr)
        optimizer = optim.AdamW(parameters, lr=self.hparams.max_lr, weight_decay=self.hparams.weight_decay)
        warmup_steps = self.hparams.warmup_epochs * self.hparams.num_train_batches
        warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=self.hparams.min_lr/self.hparams.max_lr, total_iters=warmup_steps)
        cooldown_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(self.hparams.epochs-self.hparams.warmup_epochs)*self.hparams.num_train_batches, eta_min=self.hparams.min_lr)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cooldown_scheduler], milestones=[warmup_steps])

        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}

    def forward(self, images):
        return self.model(images)

    def general_step(self, batch, batch_idx, mode, dataloader_idx=0):
        images, target = batch # extracts the images and targets for the batch
        prediction = self(images) # forward pass
        batch_size = images['Sentinel2']['data'].shape[0] # number of items in batch

        if self.hparams.task == 'biomass':
            valid_mask = target != biomass_no_data_value # mask for the NaN pixels in the target

            if self.hparams.model == 'dinov2':
                prediction = prediction[:, :, 6:-6, 6:-6] # removes the padding added for DINOv2

            prediction = prediction[valid_mask]
            target = target[valid_mask]

        if mode == 'train' or mode == 'val':
            loss = self.criterion(prediction, target) # computes the loss
            self.log(f'{mode.capitalize()} loss', loss, batch_size=batch_size) # logs the loss

        if self.hparams.task == 'species':
            prediction = torch.sigmoid(prediction) # converts logits to probabilities
            target = target.long()

        if mode == 'train' or mode == 'val':
            metrics = getattr(self, f'{mode}_metrics')
        else: # test mode
            split = 'random_test' if dataloader_idx == 0 else 'geographic_test'
            metrics = getattr(self, f'{split}_metrics')

            if not hasattr(self, f'{split}_predictions'):
                setattr(self, f'{split}_predictions', [])
                setattr(self, f'{split}_targets', [])

            getattr(self, f'{split}_predictions').append(prediction.detach().cpu())
            getattr(self, f'{split}_targets').append(target.detach().cpu())

        metrics(prediction, target) # calculates the metrics
        self.log_dict(metrics, batch_size=batch_size) # logs the metrics

        if batch_idx == 0: # if we are on the first batch
            if mode == 'train' or mode == 'val':
                self._log_images(images['Sentinel2']['data'].cpu().numpy()[:, [3,2,1]].astype(float), mode)
            else:
                self._log_images(images['Sentinel2']['data'].cpu().numpy()[:, [3,2,1]].astype(float), split)

        if mode == 'train':
            return loss

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch=batch, batch_idx=batch_idx, mode='train')

        return loss

    def validation_step(self, batch, batch_idx):
        self.general_step(batch=batch, batch_idx=batch_idx, mode='val')

    def test_step(self, batch, batch_idx, dataloader_idx):
        self.general_step(batch=batch, batch_idx=batch_idx, mode='test', dataloader_idx=dataloader_idx)

    def on_test_end(self):
        self._plot_predictions_vs_targets('random_test')
        self._plot_predictions_vs_targets('geographic_test')

    def _log_images(self, images, mode):
        images = np.array([np.stack(utils.normalize(image), axis=-1) for image in images])
        # images = np.ma.masked_equal(images, self.hparams.nodata_value)
        num_images = min(len(images), num_logged_images)
        fig, axes = plt.subplots(1, num_images, figsize=(num_images*4, 4))

        for i in range(num_images):
            axes[i].imshow(images[i])
            axes[i].axis('off')

        plt.tight_layout()
        os.makedirs('figures', exist_ok=True)
        plt.savefig(f'figures/{mode}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        if wandb.run is not None:
            wandb.log({f'{mode.replace("_", " ").capitalize()} images (RGB)': wandb.Image(f'figures/{mode}.png')})

    def _plot_predictions_vs_targets(self, split):
        # concatenate all batches
        predictions = torch.cat(getattr(self, f'{split}_predictions')).numpy().flatten()
        targets = torch.cat(getattr(self, f'{split}_targets')).numpy().flatten()

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(targets, predictions, alpha=0.5, s=5) # plots predictions vs. targets

        # Add 1:1 line
        min_val = min(np.min(predictions), np.min(targets))
        max_val = max(np.max(predictions), np.max(targets))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 line')

        # Get metrics
        metrics = getattr(self, f'{split}_metrics').compute()
        r2 = metrics[f'{split.replace("_", " ").capitalize()} R2'].item()
        rmse = metrics[f'{split.replace("_", " ").capitalize()} RMSE'].item()
        mae = metrics[f'{split.replace("_", " ").capitalize()} MAE'].item()
        me = metrics[f'{split.replace("_", " ").capitalize()} ME'].item()
        metrics_text = f'R²: {r2:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nME: {me:.4f}'
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Fit a regression line
        z = np.polyfit(x=targets, y=predictions, deg=1) # linear regression (least squares polynomial fit)
        p = np.poly1d(z) # polynomial function
        ax.plot(np.sort(targets), p(np.sort(targets)), 'b-', label=f'Fit: y={z[0]:.4f}x+{z[1]:.4f}')

        # Set labels and title
        ax.set_xlabel('Target', fontsize=14)
        ax.set_ylabel('Prediction', fontsize=14)
        ax.set_title(f'{self.hparams.task.replace("_", " ").capitalize().replace("ph", "pH")} {self.hparams.model.capitalize().replace("Mpmae", "MPMAE").replace("_", "-").replace("mme", "MME").replace("imagenet", "ImageNet").replace("Resnet", "ResNet")} {self.hparams.tuning_mode.upper()} {split.replace("_", " ")} set', fontsize=16)
        ax.legend()
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.7)
        fig.tight_layout()
        fig.savefig(f'figures/{split}_scatter_plot.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        if wandb.run is not None:
            wandb.log({f'{split.replace("_", " ").capitalize()} scatter plot': wandb.Image(f'figures/{split}_scatter_plot.png')})
