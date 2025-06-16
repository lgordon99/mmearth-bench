# ============================================== IMPORTS ============================================== #

from collections import OrderedDict
from convnextv2 import ConvNeXtV2
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
        print("Weights of {} not initialized from pretrained model: {}".format(model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(model.__class__.__name__, ignore_missing_keys))
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

class ResNet50(nn.Module):
    def __init__(self, num_classes, pixelwise, pretrained):
        super(ResNet50, self).__init__()

        self.model = resnet50(weights='DEFAULT' if pretrained else None)

        if pixelwise:
            self.model = nn.Sequential(*list(self.model.children())[:-2]) # remove the pooling and classifier layers
        else:
            self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=num_classes)

    def forward(self, images):
        return self.model(images[:, [3,2,1], :, :])

class MPMAE(nn.Module):
    def __init__(self, num_classes, pixelwise, pretrained):
        super(MPMAE, self).__init__()

        self.pixelwise = pixelwise
        self.model = ConvNeXtV2(num_classes=num_classes)

        if pixelwise:
            self.model.head = nn.Identity() # removes the classifier layer

        if pretrained:
            checkpoint_path = '/n/davies_lab/Users/luciagordon/mmearth-bench/all_mod_atto_1M_64_uncertainty_56-8.pth' # Vishal's checkpoint
            load_custom_checkpoint(self.model, checkpoint_path) # freezing and unfreezing is done in this function

    def _forward_features(self, x):
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
        if self.pixelwise:
            return self._forward_features(images)
        else:
            return self.model(images)

class DINOv2(nn.Module):
    def __init__(self, num_classes, pixelwise, pretrained):
        super(DINOv2, self).__init__()

        self.pixelwise = pixelwise
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        self.model.interpolate_pos_encoding = self._deterministic_interpolate_pos_encoding

        if not pixelwise:
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
        images = images[:, [3,2,1], :, :]

        if self.pixelwise:
            features = self.model.forward_features(images)['x_norm_patchtokens'].permute(0, 2, 1)

            return features.reshape(features.shape[0], features.shape[1], int(np.sqrt(features.shape[2])), int(np.sqrt(features.shape[2])))
        else:
            return self.model(images)

class LinearDecoder(nn.Module):
    def __init__(self, feature_dim):
        super(LinearDecoder, self).__init__()

        self.model = nn.Sequential(nn.Upsample(size=128, mode='bilinear'),
                                   nn.Conv2d(in_channels=feature_dim, out_channels=1, kernel_size=1))

    def forward(self, images):
        return self.model(images)

class EncoderDecoder(nn.Module):
    def __init__(self, model_class_name, pixelwise, num_classes, pretrained):
        super(EncoderDecoder, self).__init__()

        self.pixelwise = pixelwise
        self.model = globals()[model_class_name](num_classes, pixelwise, pretrained)

        if pixelwise:
            if model_class_name == 'ResNet50':
                feature_dim = 2048
            elif model_class_name == 'DINOv2':
                feature_dim = 384
            elif model_class_name == 'MPMAE':
                feature_dim = 320

            self.decoder = LinearDecoder(feature_dim) # creates the decoder with the number of features

    def forward(self, images):
        if self.pixelwise:
            return self.decoder(self.model(images)) # applies the decoder to the features
        else:
            return self.model(images)

class Model(LightningModule):
    def __init__(self, task, model, adaptation_mode, decay_factor, max_lr, weight_decay, warmup_epochs, num_train_batches, min_lr, epochs, nodata_value):
        super().__init__()

        self.save_hyperparameters()
        self.configure_models()
        self.configure_metrics()

        if task == 'species': # multi-label classification
            self.criterion = nn.BCEWithLogitsLoss()
        else: # regression
            self.criterion = nn.MSELoss()

    def configure_models(self):
        pixelwise = True if self.hparams.task == 'biomass' else False
        num_classes = 100 if self.hparams.task == 'species' else 1

        if self.hparams.model == 'resnet50':
            self.model = EncoderDecoder(model_class_name='ResNet50', pixelwise=pixelwise, num_classes=num_classes, pretrained=False)
        elif self.hparams.model == 'resnet50_imagenet':
            self.model = EncoderDecoder(model_class_name='ResNet50', pixelwise=pixelwise, num_classes=num_classes, pretrained=True)
        elif self.hparams.model == 'dinov2':
            self.model = EncoderDecoder(model_class_name='DINOv2', pixelwise=pixelwise, num_classes=num_classes, pretrained=None)
        elif self.hparams.model == 'mpmae':
            self.model = EncoderDecoder(model_class_name='MPMAE', pixelwise=pixelwise, num_classes=num_classes, pretrained=False)
        elif self.hparams.model == 'mpmae_mmearth':
            self.model = EncoderDecoder(model_class_name='MPMAE', pixelwise=pixelwise, num_classes=num_classes, pretrained=True)

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
        if self.hparams.adaptation_mode == 'lp':
            # freeze all parameters
            for param in self.model.parameters():
                param.requires_grad = False

            # unfreeze the final or decoder layers
            if 'resnet' in self.hparams.model:
                parameters_to_unfreeze = self.model.model.model.fc.parameters() # final layer
            # elif 'pixel_reg' in self.hparams.model:
            #     children_to_unfreeze = ['norm', 'head', 'upsample_layers', 'initial_conv_upsample'] # decoder parts
            #     parameters_to_unfreeze = [p for name, module in self.model.named_modules() for child_name in children_to_unfreeze if child_name in name for p in module.parameters()]
            elif 'dino' in self.hparams.model or 'mpmae' in self.hparams.model:
                parameters_to_unfreeze = self.model.model.model.head.parameters() # final layer

            for param in parameters_to_unfreeze:
                param.requires_grad = True

        if self.hparams.adaptation_mode == 'llrd':
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

            layer_names.reverse()
            print(f'{len(layer_names)} layer groups')
            print(layer_names)
            parameters = []

            for i, name in enumerate(layer_names):
                learning_rate = max_lr * self.hparams.decay_factor ** i
                print(f'{name}: {learning_rate}')
                parameters += [{'params': [p for n, p in self.model.model.model.named_parameters() if n == name or (len(n.split(name)) > 1 and n.startswith(name) and n.split(name)[1][0] == '.') or (name == 'embedding_tokens' and (n == 'cls_token' or n == 'pos_embed' or n == 'register_tokens' or n == 'mask_token'))],
                                'lr': learning_rate,
                                'name': name}]

            assert sum(p.numel() for p in self.model.parameters()) == sum(p.numel() for group in parameters for p in group['params'])

            return parameters

        return self.model.parameters() # fine-tuning

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

        if self.hparams.model == 'dinov2':
            images = torch.nn.functional.pad(images, (6, 6, 6, 6), mode='constant', value=self.hparams.nodata_value) # DinoV2 requires the image size to be divisible by 14

        prediction = self(images) # forward pass
        batch_size = images.shape[0] # number of items in batch

        if self.hparams.task == 'biomass':
            valid_mask = target != self.hparams.nodata_value # mask for the NaN pixels in the target
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
                self._log_images(images.cpu().numpy()[:, [3,2,1]].astype(float), mode)
            else:
                self._log_images(images.cpu().numpy()[:, [3,2,1]].astype(float), split)

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
        images = np.ma.masked_equal(images, self.hparams.nodata_value)
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
        ax.set_title(f'{self.hparams.task.replace("_", " ").capitalize().replace("ph", "pH")} {self.hparams.model.capitalize().replace("Mpmae", "MPMAE").replace("_", "-").replace("mme", "MME").replace("imagenet", "ImageNet").replace("Resnet", "ResNet")} {self.hparams.adaptation_mode.upper()} {split.replace("_", " ")} set', fontsize=16)
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
