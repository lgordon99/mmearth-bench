# ============================================== IMPORTS ============================================== #

from collections import OrderedDict
from convnextv2 import convnextv2_atto
from convnextv2_unet import convnextv2_unet_atto
from lightning.pytorch import LightningModule
from timm.models.layers import trunc_normal_
from torchmetrics import MetricCollection, Recall
from torchmetrics.classification import MultilabelRecall, MultilabelAveragePrecision
from torchmetrics.regression import MeanSquaredError
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

# ============================================== FUNCTIONS ============================================== #

def remap_checkpoint_keys(ckpt):
    # function adapted from https://github.com/vishalned/MMEarth-train/blob/main/helpers.py

    new_ckpt = OrderedDict()

    for k, v in ckpt.items():
        if k.startswith('encoder'):
            k = ".".join(k.split(".")[1:])  # remove encoder in the name
        if k.endswith('kernel'):
            k = ".".join(k.split(".")[:-1])  # remove kernel in the name
            new_k = k + '.weight'
            if len(v.shape) == 3:  # resahpe standard convolution
                kv, in_dim, out_dim = v.shape
                ks = int(math.sqrt(kv))
                new_ckpt[new_k] = (v.permute(2, 1, 0).reshape(out_dim, in_dim, ks, ks).transpose(3, 2))
            elif len(v.shape) == 2:  # reshape depthwise convolution
                kv, dim = v.shape
                ks = int(math.sqrt(kv))
                new_ckpt[new_k] = (v.permute(1, 0).reshape(dim, 1, ks, ks).transpose(3, 2))
            continue
        elif 'ln' in k or 'linear' in k:
            k = k.split('.')
            k.pop(-2)  # remove ln and linear in the name
            new_k = '.'.join(k)
        else:
            new_k = k

        new_ckpt[new_k] = v

    # reshape grn affine parameters and biases
    for k, v in new_ckpt.items():
        if k.endswith('bias') and len(v.shape) != 1:
            new_ckpt[k] = v.reshape(-1)
        elif 'grn' in k:
            new_ckpt[k] = v.unsqueeze(0).unsqueeze(1)

    return new_ckpt

def load_state_dict(model, state_dict, ignore_missing="relative_position_index", quit=False):
    # function adapted from https://github.com/vishalned/MMEarth-train/blob/main/helpers.py

    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()

    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)

        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(model)

    warn_missing_keys = []
    ignore_missing_keys = []

    for key in missing_keys:
        keep_flag = True

        for ignore_key in ignore_missing.split("|"):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print(
            "Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys
            )
        )
    if len(unexpected_keys) > 0:
        print(
            "Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys
            )
        )
    if len(ignore_missing_keys) > 0:
        print(
            "Ignored weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, ignore_missing_keys
            )
        )
    if len(error_msgs) > 0:
        print("\n".join(error_msgs))

def load_custom_checkpoint(model, checkpoint_path):
    # function adapted from https://github.com/vishalned/MMEarth-train/blob/main/helpers.py

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print(f'Load pre-trained checkpoint from: {checkpoint_path}')
    checkpoint_model = checkpoint['model'] if 'model' in checkpoint else checkpoint
    state_dict = model.state_dict()

    for k in ['head.weight', 'head.bias']:
        if (k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape):
            print(f'Removing key {k} from pretrained checkpoint')
            del checkpoint_model[k]

    # remove decoder weights
    checkpoint_model_keys = list(checkpoint_model.keys())

    for k in checkpoint_model_keys:
        if 'decoder' in k or 'mask_token' in k or 'proj' in k or 'pred' in k:
            print(f'Removing key {k} from pretrained checkpoint')
            del checkpoint_model[k]

    checkpoint_model = remap_checkpoint_keys(checkpoint_model)
    load_state_dict(model, checkpoint_model)

    # manually initialize fc layer
    trunc_normal_(model.head.weight, std=2e-5)
    torch.nn.init.constant_(model.head.bias, 0.)

    return model

def center_crop(array, crop_size=112):
    height, width = array.shape[-2:]
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    bottom = top + crop_size
    right = left + crop_size

    return array[..., top:bottom, left:right]

# ============================================== CLASSES ============================================== #

class ResNet50(nn.Module):
    def __init__(self, num_classes, weights):
        super(ResNet50, self).__init__()
        self.model = resnet50(weights=weights)
        num_features = self.model.fc.in_features # input to final layer
        self.model.fc = nn.Linear(in_features=num_features, out_features=num_classes)

    def forward(self, images):
        return self.model(images[:, [3,2,1], :, :])

class ResNet50Sentinel2(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50Sentinel2, self).__init__()
        self.model = resnet50(weights='DEFAULT')
        num_bands = 12
        original_weights = self.model.conv1.weight.clone()
        self.model.conv1 = nn.Conv2d(in_channels=num_bands,
                                     out_channels=self.model.conv1.out_channels,
                                     kernel_size=self.model.conv1.kernel_size,
                                     stride=self.model.conv1.stride,
                                     padding=self.model.conv1.padding,
                                     bias=self.model.conv1.bias)

        with torch.no_grad():
            self.model.conv1.weight[:, 3] = original_weights[:, 0] # R
            self.model.conv1.weight[:, 2] = original_weights[:, 1] # G
            self.model.conv1.weight[:, 1] = original_weights[:, 2] # B
            mean_original_weights = torch.mean(original_weights, dim=1)
            self.model.conv1.weight[:, 0] = mean_original_weights
            self.model.conv1.weight[:, 4:] = mean_original_weights.unsqueeze(1).expand(-1, self.model.conv1.weight.shape[1]-4, -1, -1)
            self.model.conv1.weight *= 3/num_bands

        num_features = self.model.fc.in_features # input to final layer
        self.model.fc = nn.Linear(in_features=num_features, out_features=num_classes)

    def forward(self, images):
        return self.model(images)

class ConvNextV2(nn.Module):
    def __init__(self, num_classes, mmearth):
        super(ConvNextV2, self).__init__()

        self.model = convnextv2_atto(patch_size=8, # patch size used during pretraining
                                     img_size=56, # patch size used during pretraining
                                     in_chans=12, # number of Sentinel-2 bands
                                     num_classes=num_classes)

        if mmearth:
            checkpoint_path = '/n/davies_lab/Users/luciagordon/mmearth-bench/all_mod_atto_1M_64_uncertainty_56-8.pth' # Vishal's checkpoint
            self.model = load_custom_checkpoint(self.model, checkpoint_path) # freezing and unfreezing is done in this function

    def forward(self, images):
        return self.model(images)

class ConvnextV2Unet(nn.Module):
    def __init__(self, mmearth):
        super(ConvnextV2Unet, self).__init__()

        self.model = convnextv2_unet_atto(patch_size=8, # patch size used during pretraining
                                          img_size=56, # patch size used during pretraining
                                          in_chans=12, # number of Sentinel-2 bands
                                          num_classes=1) # regression

        if mmearth:
            checkpoint_path = '/n/davies_lab/Users/luciagordon/mmearth-bench/all_mod_atto_1M_64_uncertainty_56-8.pth' # Vishal's checkpoint
            self.model = load_custom_checkpoint(self.model, checkpoint_path) # freezing and unfreezing is done in this function

    def forward(self, images):
        return self.model(images)

class Dinov2(nn.Module):
    def __init__(self, num_classes):
        super(Dinov2, self).__init__()

        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        self.model.head = nn.Linear(in_features=384, out_features=num_classes)
        # self.head = torch.load('/n/davies_lab/Users/luciagordon/mmearth-bench/dinov2_vits14_ade20k_ms_head.pth')

    def forward(self, images):
        return self.model(images[:, [3,2,1], :, :])

class Model(LightningModule):
    def __init__(self, task, model, adaptation_mode, decay_factor, max_lr, weight_decay, epochs, min_lr, warmup_epochs, num_train_batches):
        super().__init__()

        self.save_hyperparameters()
        self.configure_models()
        self.configure_metrics()

        if task == 'species': # multi-label classification
            self.criterion = nn.BCEWithLogitsLoss()
        else: # regression
            self.criterion = nn.MSELoss()

    def configure_models(self):
        num_classes = 100 if self.hparams.task == 'species' else 1

        if self.hparams.model == 'resnet50':
            self.model = ResNet50(num_classes=num_classes, weights=None)
        elif self.hparams.model == 'resnet50_imagenet':
            self.model = ResNet50(num_classes=num_classes, weights='DEFAULT')
        elif self.hparams.model == 'resnet_sentinel2':
            self.model = ResNet50Sentinel2(num_classes)
        elif self.hparams.model == 'unet':
            self.model = smp.Unet(encoder_name='resnet50',
                                  encoder_weights='imagenet',
                                  in_channels=12)
        elif self.hparams.model == 'mpmae_mmearth':
            self.model = ConvNextV2(num_classes=num_classes, mmearth=True)
            # self.model = torch.hub.load('vishalned/mmearth-train', 'MPMAE', trust_repo=True, num_classes=num_classes)
        elif self.hparams.model == 'mpmae':
            self.model = ConvNextV2(num_classes=num_classes, mmearth=False)
        elif self.hparams.model == 'mpmae_pixel_reg_mmearth':
            self.model = ConvnextV2Unet(mmearth=True)
        elif self.hparams.model == 'mpmae_pixel_reg':
            self.model = ConvnextV2Unet(mmearth=False)
        elif self.hparams.model == 'dinov2':
            self.model = Dinov2(num_classes)
        elif self.hparams.model == 'anysat':
            self.model = torch.hub.load('gastruc/anysat', 'anysat', pretrained=True, flash_attn=False)

    def configure_metrics(self):
        if self.hparams.task == 'species':
            num_labels = 100
            metrics = MetricCollection({'Recall': MultilabelRecall(num_labels),
                                        'MAP': MultilabelAveragePrecision(num_labels)})
        else:
            metrics = MetricCollection({'RMSE': MeanSquaredError(squared=False)})

        # self.train_metrics = metrics.clone(prefix='Train ')
        # self.val_metrics = metrics.clone(prefix='Val ')
        # self.test_metrics = metrics.clone(prefix='Test ')

        for split in ['train', 'val', 'random_test', 'geographic_test']:
            setattr(self, f'{split}_metrics', metrics.clone(prefix=f'{split.replace("_", " ").capitalize()} '))

    def configure_parameters(self, max_lr):
        if self.hparams.adaptation_mode == 'llrd':
            layer_names = []

            for name, _ in self.model.model.named_parameters():
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

                if layer_name not in layer_names:
                    layer_names.append(layer_name)

            layer_names.reverse()
            print(f'{len(layer_names)} layer groups')
            print(layer_names)
            parameters = []

            for i, name in enumerate(layer_names):
                learning_rate = max_lr * self.hparams.decay_factor ** i
                print(f'{name}: {learning_rate}')
                parameters += [{'params': [p for n, p in self.model.model.named_parameters() if n.startswith(name)],
                                'lr': learning_rate,
                                'name': name}]

            return parameters

        if self.hparams.adaptation_mode != 'ft':
            # freeze all parameters
            for param in self.model.parameters():
                param.requires_grad = False

            # unfreeze the final or decoder layers
            if 'resnet' in self.hparams.model:
                parameters_to_unfreeze = self.model.model.fc.parameters() # final layer
            elif 'pixelwise_regression' in self.hparams.model:
                children_to_unfreeze = ['norm', 'head', 'upsample_layers', 'initial_conv_upsample'] # decoder parts
                parameters_to_unfreeze = [p for name, module in self.model.named_modules() for child_name in children_to_unfreeze if child_name in name for p in module.parameters()]
            elif 'mpmae' in self.hparams.model:
                parameters_to_unfreeze = self.model.model.head.parameters() # final layer
            elif 'dino' in self.hparams.model:
                parameters_to_unfreeze = self.model.head.parameters() # final layer
                pdb.set_trace()

            for param in parameters_to_unfreeze:
                param.requires_grad = True

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
        images, target = batch
        prediction = self(images)
        batch_size = images.shape[0]

        if self.hparams.task == 'biomass':
            valid_mask = target != -9999
            prediction = prediction[valid_mask]
            target = target[valid_mask]

        if mode == 'train' or mode == 'val':
            loss = self.criterion(prediction, target)
            self.log(f'{mode.capitalize()} loss', loss, batch_size=batch_size)

        if self.hparams.task == 'species':
            prediction = torch.sigmoid(prediction) # converts logits to probabilities
            target = target.long()

        if mode == 'train' or mode == 'val':
            metrics = getattr(self, f'{mode}_metrics')
        else:
            split = 'random_test' if dataloader_idx == 0 else 'geographic_test'
            metrics = getattr(self, f'{split}_metrics')

        metrics(prediction, target)
        self.log_dict(metrics, batch_size=batch_size)

        if batch_idx == 0: # if we are on the first batch
            if mode == 'train' or mode == 'val':
                self._log_images(images.cpu().numpy()[:, [3,2,1]].astype(float), mode)
                wandb.log({mode.capitalize(): wandb.Image(f'figures/{mode}.png')})
            else:
                self._log_images(images.cpu().numpy()[:, [3,2,1]].astype(float), split)
                wandb.log({split.replace('_', ' ').capitalize(): wandb.Image(f'figures/{split}.png')})

        if mode == 'train':
            return loss

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch=batch, batch_idx=batch_idx, mode='train')
        # images, target = batch
        # prediction = self(images)
        # batch_size = images.shape[0]

        # if self.hparams.task == 'biomass':
        #     valid_mask = target != -9999
        #     prediction = prediction[valid_mask]
        #     target = target[valid_mask]

        # loss = self.criterion(prediction, target)

        # self.log('Train loss', loss, batch_size=batch_size)

        # if self.hparams.task == 'species':
        #     prediction = torch.sigmoid(prediction) # converts logits to probabilities
        #     target = target.long()

        # self.train_metrics(prediction, target)
        # self.log_dict(self.train_metrics, batch_size=batch_size)

        # if batch_idx == 0: # if we are on the first batch
        #     self._log_images(images.cpu().numpy()[:, [3,2,1]].astype(float), 'training')
        #     wandb.log({'Training': wandb.Image('figures/training.png')})

        return loss

    def validation_step(self, batch, batch_idx):
        self.general_step(batch=batch, batch_idx=batch_idx, mode='val')
        # images, target = batch
        # prediction = self(images)
        # batch_size = images.shape[0]

        # if self.hparams.task == 'biomass':
        #     valid_mask = target != -9999
        #     prediction = prediction[valid_mask]
        #     target = target[valid_mask]

        # loss = self.criterion(prediction, target)

        # self.log('Val loss', loss, batch_size=batch_size)

        # if self.hparams.task == 'species':
        #     prediction = torch.sigmoid(prediction) # converts logits to probabilities
        #     target = target.long()

        # self.val_metrics(prediction, target)
        # self.log_dict(self.val_metrics, batch_size=batch_size)

        # if batch_idx == 0: # if we are on the first batch
        #     self._log_images(images.cpu().numpy()[:, [3,2,1]].astype(float), 'validation')
        #     wandb.log({'Validation': wandb.Image('figures/validation.png')})

    def test_step(self, batch, batch_idx, dataloader_idx):
        self.general_step(batch=batch, batch_idx=batch_idx, mode='test', dataloader_idx=dataloader_idx)
        # images, target = batch
        # prediction = self(images)
        # batch_size = images.shape[0]

        # if self.hparams.task == 'biomass':
        #     valid_mask = target != -9999
        #     prediction = prediction[valid_mask]
        #     target = target[valid_mask]
        # elif self.hparams.task == 'species':
        #     prediction = torch.sigmoid(prediction) # converts logits to probabilities
        #     target = target.long()

        # split = 'random_test' if dataloader_idx == 0 else 'geographic_test'
        # metrics = getattr(self, f'{split}_metrics')
        # metrics(prediction, target)
        # self.log_dict(metrics, batch_size=batch_size)

        # self.test_metrics(prediction, target)
        # self.log_dict(self.test_metrics, batch_size=batch_size)

        # if batch_idx == 0: # if we are on the first batch
        #     self._log_images(images.cpu().numpy()[:, [3,2,1]].astype(float), split)
        #     wandb.log({split.replace('_', ' ').capitalize(): wandb.Image(f'figures/{split}.png')})

    def on_train_epoch_start(self):
        if self.current_epoch == 50 and self.hparams.adaptation_mode == 'two_stage':
            optimizer = self.trainer.optimizers[0]

            num_params_in_optimizer = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
            print("Total number of parameters in optimizer:", num_params_in_optimizer)
            num_params = sum(p.numel() for p in self.model.parameters())
            print("Total number of parameters:", num_params)
            print(num_params_in_optimizer == num_params)
            num_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print("Total number of trainable parameters:", num_trainable_params)

            for param in self.model.parameters():
                param.requires_grad = True

            num_params_in_optimizer = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
            print("Total number of parameters in optimizer:", num_params)
            num_params = sum(p.numel() for p in self.model.parameters())
            print("Total number of parameters:", num_params)
            print(num_params_in_optimizer == num_params)
            num_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print("Total number of trainable parameters:", num_trainable_params)

    def _log_images(self, images, mode):
        images = np.array([np.stack(utils.normalize(image), axis=-1) for image in images])
        num_images = min(len(images), num_logged_images)
        fig, axes = plt.subplots(1, num_images, figsize=(num_images*4, 4))

        for i in range(num_images):
            axes[i].imshow(images[i])
            axes[i].axis('off')

            if i == int(num_images / 2):
                axes[i].set_title('RGB Images', fontsize=fontsize, pad=pad)

        plt.tight_layout()
        os.makedirs('figures', exist_ok=True)
        plt.savefig(f'figures/{mode}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
