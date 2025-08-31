# ============================================== IMPORTS ============================================== #

from convnextv2 import ConvNeXtV2, load_custom_checkpoint
from lightning.pytorch import LightningModule
from tabulate_results import get_best_run_in_sweep
from torchmetrics import Metric, MetricCollection, Recall
from torchmetrics.classification import MultilabelRecall, MultilabelAveragePrecision
from torch.func import functional_call
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score
from torchvision.models import resnet50
import copy
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
entity = os.environ['ENTITY']
project = os.environ['PROJECT']
no_data_values = utils.read_json(f'{data_dir_path}/no_data_values.json')
biomass_no_data_value = -9999
architecture_properties = {'ResNet50': {'num_image_channels': 3, 'embedding_dim': 2048}, 'DINOv2': {'num_image_channels': 3, 'embedding_dim': 384}, 'MPMAE': {'num_image_channels': 12, 'embedding_dim': 320}}

# ============================================== FUNCTIONS ============================================== #

def get_run_id(task, architecture, adaptation_mode, tuning_mode):
    name = '_'.join([task, architecture, adaptation_mode, tuning_mode])
    best_run = get_best_run_in_sweep(name, data_dir_path)

    return best_run["ID"], name

def get_state_dict(run_id, name):
    run = wandb.Api().run(f'{entity}/{project}/{run_id}')

    for artifact in run.logged_artifacts():
        if 'best' in artifact.aliases:
            artifact.download(f'/tmp/{name}')

    ckpt = torch.load(f'/tmp/{name}/model.ckpt')

    return ckpt['state_dict']

def get_log_name(mode, dataloader_idx):
    if mode == 'test':
        split = 'random_test' if dataloader_idx == 0 else 'geographic_test'
        name = split.capitalize().replace('_', ' ')
    else:
        name = mode.capitalize()

    return name

# ============================================== CLASSES ============================================== #

class MeanError(Metric):
    def __init__(self):
        super().__init__()

        self.add_state('error_sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, prediction, target):
        self.error_sum += torch.sum(prediction - target)
        self.count += target.numel() # counts the number of elements

    def compute(self):
        return self.error_sum / self.count

class AdaptationImprovement(Metric):
    def __init__(self):
        super().__init__()

        self.add_state('initial_loss_sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('final_loss_sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        # self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, initial_loss, final_loss):
        self.initial_loss_sum += initial_loss.sum()
        self.final_loss_sum += final_loss.sum()
        # self.count += initial_loss.numel() # counts the number of elements

    def compute(self):
        return (self.initial_loss_sum - self.final_loss_sum) / self.initial_loss_sum * 100 # percent improvement

                # initial_predictions = torch.cat(getattr(self, f'{split}_iteration_predictions'), dim=1)[0]
                # initial_loss = nn.MSELoss()(initial_predictions, targets) if self.hparams.task != 'species' else nn.BCEWithLogitsLoss()(initial_predictions, targets)
                # self.log(f'{get_log_name(mode, dataloader_idx)} initial loss', initial_loss) # logs the loss prior to adaptation
                # adaptation_improvement = (initial_loss - loss) / initial_loss * 100 # larger is better
                # self.log(f'{get_log_name(mode, dataloader_idx)} loss adaptation improvement %', adaptation_improvement, batch_size=batch_size)

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
        self.model.fc = nn.Identity()

    def forward(self, images): # doesn't call the avgpool, flatten, and fc layers
        x = images['Sentinel2']['data'][:, [3,2,1], :, :] # extracts the RGB bands
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        return x

class TaskDecoder(nn.Module):
    def __init__(self, architecture, pixelwise, adaptation_mode, num_classes):
        super().__init__()

        self.architecture = architecture
        self.num_classes = num_classes

        if architecture == 'ResNet50':
            self.decoder = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                         nn.Flatten(),
                                         nn.Linear(in_features=architecture_properties[architecture]['embedding_dim'], out_features=num_classes)) # infers in_features from input shape

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

        self.task_modality_decoder = LinearDecoder(image_size=128,
                                                   num_image_channels=num_image_channels,
                                                   embedding_dim=embedding_dim,
                                                   out_channels=922) # 922 is the total number of bands in the task modalities excluding the NaN bands in the categorical modalities
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

        self.categorical_modalities = ['DynamicWorld', 'ESA_WorldCover', 'biome', 'ecoregion']
        self.loss_functions = {modality: nn.CrossEntropyLoss(ignore_index=no_data_values[modality], reduction='none') if modality in self.categorical_modalities else nn.MSELoss(reduction='none') for modality in no_data_values.keys()}

    def forward(self, modality_reconstructions, images):
        modality_reconstruction_losses = {}

        for modality, reconstruction in modality_reconstructions.items():
            target = images[modality]['data']

            if modality in self.categorical_modalities:
                if len(target.shape) > 1:
                    target = target.squeeze(1) # removes the channel dimension

                target = target.long() # casts to torch.int64, the type for class indices in CrossEntropyLoss

            reconstruction_loss = self.loss_functions[modality](reconstruction, target)

            if 'valid_mask' in images[modality].keys(): # for modalities that can have NaNs
                valid_mask = images[modality]['valid_mask']

                if len(reconstruction_loss.shape) > 1: # for modalities with either channel or pixel dimensions
                    masked_reconstruction_loss = reconstruction_loss.masked_fill(~valid_mask, 0) # sets the no data pixels to zero
                    dims_to_reduce = tuple(range(1, len(masked_reconstruction_loss.shape))) # channel and pixel dimensions as present
                    sum_ = masked_reconstruction_loss.sum(dim=dims_to_reduce) # numerator in average
                    count = valid_mask.sum(dim=dims_to_reduce) # number of valid pixels in each image
                    mean = sum_ / count.clamp_min(1) # prevents division by 0
                    modality_reconstruction_losses[modality] = mean.masked_fill(count == 0, float('nan')) # sets the reconstruction loss to NaN for the image if there are no valid pixels for the modality
                else: # for biome and ecoregion
                    modality_reconstruction_losses[modality] = reconstruction_loss.masked_fill(~valid_mask, float('nan')) # sets any no data pixels to NaN

                # masked_reconstruction_loss = reconstruction_loss.masked_fill(~valid_mask, float('nan')) # sets the no data pixels to NaN
                # modality_reconstruction_losses[modality] = torch.nanmean(masked_reconstruction_loss, dim=tuple(range(1, len(masked_reconstruction_loss.shape)))) if len(masked_reconstruction_loss.shape) > 1 else masked_reconstruction_loss # computes the mean loss per image across all but the batch dimension, ignoring NaNs
            else: # geolocation and month
                modality_reconstruction_losses[modality] = torch.mean(reconstruction_loss, dim=1) # computes the mean loss per image across the channel dimension

        return modality_reconstruction_losses

class TaskModalityDecoderLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.modality_reconstruction_loss_calculator = ModalityReconstructionLossCalculator()

    def forward(self, modality_reconstructions, images):
        modality_reconstruction_losses = self.modality_reconstruction_loss_calculator(modality_reconstructions, images)
        mean_loss = torch.stack(list(modality_reconstruction_losses.values()), dim=1).nanmean() # computes the mean loss across all modalities, ignoring NaNs

        return mean_loss

class SurrogateLossNetwork(nn.Module):
    def __init__(self, average_over_batch):
        super().__init__()

        self.average_over_batch = average_over_batch
        # self.surrogate_loss_network = nn.Sequential(nn.Linear(24, 64),
        #                                             nn.GELU(),
        #                                             nn.Linear(64, 64),
        #                                             nn.GELU(),
        #                                             nn.Linear(64, 1),
        #                                             nn.Softplus()) # ensures the output is non-negative
        in_features = 24 # 12 modalities, each with a reconstruction loss and a mask value
        self.surrogate_loss_network = nn.Linear(in_features, 1)

        with torch.no_grad():
            self.surrogate_loss_network.weight.fill_(1.0 / in_features) # initializes all weights to 1/in_features
            self.surrogate_loss_network.bias.zero_() # initializes bias to 0

    def forward(self, modality_reconstruction_losses):
        stacked_modality_reconstruction_losses = torch.stack(list(modality_reconstruction_losses.values()), dim=1)
        # print('min', stacked_modality_reconstruction_losses.min().item(), 'max', stacked_modality_reconstruction_losses.max().item())
        existing_modality_mask = (~stacked_modality_reconstruction_losses.isnan()).float() # creates a mask for existing modalities (not NaN)
        filled_modality_reconstruction_losses = torch.nan_to_num(stacked_modality_reconstruction_losses, nan=0.0)
        # print('min', filled_modality_reconstruction_losses.min().item(), 'max', filled_modality_reconstruction_losses.max().item())
        surrogate_loss_network_input = torch.cat([filled_modality_reconstruction_losses, existing_modality_mask], dim=1)
        surrogate_loss = self.surrogate_loss_network(surrogate_loss_network_input)

        if self.average_over_batch:
            surrogate_loss = surrogate_loss.mean()

        return surrogate_loss

class EncoderDecoder(nn.Module):
    def __init__(self, task, architecture, adaptation_mode, tuning_mode, pixelwise, num_classes, pretrained, lr):
        super().__init__()

        self.adaptation_mode = adaptation_mode
        self.pixelwise = pixelwise
        self.encoder = globals()[f'{architecture}Encoder'](pixelwise, pretrained)
        self.lr = lr
        num_image_channels = architecture_properties[architecture]['num_image_channels']
        embedding_dim = architecture_properties[architecture]['embedding_dim']

        if adaptation_mode == 'standard':
            self.task_decoder = TaskDecoder(architecture, pixelwise, adaptation_mode, num_classes)
            state_dict = get_state_dict(*get_run_id(task, architecture, 'standard', tuning_mode))
            encoder_state_dict = {key.replace('model.encoder.', ''): value for key, value in state_dict.items() if key.startswith('model.encoder')} # filters the state_dict to only include the encoder parameters
            task_decoder_state_dict = {key.replace('model.task_decoder.', ''): value for key, value in state_dict.items() if key.startswith('model.task_decoder')} # filters the state_dict to only include the decoder parameters
            self.encoder.load_state_dict(encoder_state_dict)
            self.task_decoder.load_state_dict(task_decoder_state_dict)
        elif adaptation_mode == 'multimodal':
            self.task_modality_encoder = TaskModalityEncoder()
        elif adaptation_mode == 'task_modality_decoder':
            state_dict = get_state_dict(*get_run_id(task, architecture, 'standard', tuning_mode))
            encoder_state_dict = {key.replace('model.encoder.', ''): value for key, value in state_dict.items() if key.startswith('model.encoder')} # filters the state_dict to only include the encoder parameters
            self.encoder.load_state_dict(encoder_state_dict)
            self.encoder.requires_grad_(False) # freezes the encoder
            self.task_modality_decoder = TaskModalityDecoder(num_image_channels, embedding_dim)
        elif adaptation_mode == 'tto':
            self.task_decoder = TaskDecoder(architecture, pixelwise, adaptation_mode, num_classes)
            state_dict = get_state_dict(*get_run_id(task, architecture, 'standard', tuning_mode))
            encoder_state_dict = {key.replace('model.encoder.', ''): value for key, value in state_dict.items() if key.startswith('model.encoder')} # filters the state_dict to only include the encoder parameters
            task_decoder_state_dict = {key.replace('model.task_decoder.', ''): value for key, value in state_dict.items() if key.startswith('model.task_decoder')} # filters the state_dict to only include the decoder parameters
            self.encoder.load_state_dict(encoder_state_dict)
            self.encoder.requires_grad_(False) # freezes the encoder
            self.task_decoder.load_state_dict(task_decoder_state_dict)
            self.task_decoder.requires_grad_(False) # freezes the task decoder
            self.task_modality_decoder = TaskModalityDecoder(num_image_channels, embedding_dim)
            task_modality_decoder_state_dict = get_state_dict(run_id='u7pagh3z', name='task_modality_decoder')
            task_modality_decoder_state_dict = {key.replace('model.task_modality_decoder.', ''): value for key, value in task_modality_decoder_state_dict.items() if key.startswith('model.task_modality_decoder')}
            self.task_modality_decoder.load_state_dict(task_modality_decoder_state_dict)
            self.task_modality_decoder.requires_grad_(False) # freezes the task modality decoder
            self.task_modality_decoder_loss = TaskModalityDecoderLoss()
        elif adaptation_mode == 'maml' or adaptation_mode == 'learn_loss':
            self.task_decoder = TaskDecoder(architecture, pixelwise, adaptation_mode, num_classes)
            state_dict = get_state_dict(*get_run_id(task, architecture, 'standard', tuning_mode))
            encoder_state_dict = {key.replace('model.encoder.', ''): value for key, value in state_dict.items() if key.startswith('model.encoder')} # filters the state_dict to only include the encoder parameters
            task_decoder_state_dict = {key.replace('model.task_decoder.', ''): value for key, value in state_dict.items() if key.startswith('model.task_decoder')} # filters the state_dict to only include the decoder parameters
            self.encoder.load_state_dict(encoder_state_dict)
            self.encoder.requires_grad_(False) # freezes the encoder
            self.task_decoder.load_state_dict(task_decoder_state_dict)
            self.task_decoder.requires_grad_(False) # freezes the task decoder
            self.task_modality_decoder = TaskModalityDecoder(num_image_channels, embedding_dim)
            task_modality_decoder_state_dict = get_state_dict(run_id='u7pagh3z', name='task_modality_decoder')
            task_modality_decoder_state_dict = {key.replace('model.task_modality_decoder.', ''): value for key, value in task_modality_decoder_state_dict.items() if key.startswith('model.task_modality_decoder')}
            self.task_modality_decoder.load_state_dict(task_modality_decoder_state_dict)
            self.task_modality_decoder.requires_grad_(False) # freezes the task modality decoder
            self.modality_reconstruction_loss_calculator = ModalityReconstructionLossCalculator()
            self.surrogate_loss_network = SurrogateLossNetwork(average_over_batch=adaptation_mode=='maml')

    def forward(self, images):
        if self.adaptation_mode == 'standard':
            input_embeddings = self.encoder(images)
            task_prediction = self.task_decoder(input_embeddings)

            return task_prediction
        elif self.adaptation_mode == 'multimodal':
            input_embeddings = self.encoder(images)
            modality_embeddings = self.task_modality_encoder(images)
            concatenated_embeddings = torch.cat([input_embeddings, modality_embeddings], dim=1)
            task_prediction = self.task_decoder(concatenated_embeddings)

            return task_prediction
        elif self.adaptation_mode == 'task_modality_decoder':
            self.encoder.eval() # keeps batch norm and dropout deterministic

            input_embeddings = self.encoder(images)
            modality_reconstructions = self.task_modality_decoder(input_embeddings, images)

            return modality_reconstructions
        elif self.adaptation_mode == 'tto':
            # keep batch norm and dropout deterministic
            self.encoder.eval()
            self.task_decoder.eval()
            self.task_modality_decoder.eval()

            iteration_predictions = []

            with torch.enable_grad(): # need to be able to compute gradients even during validation and testing
                encoder_parameters = {name: parameter.detach().clone().requires_grad_() for name, parameter in self.encoder.named_parameters()}

                for _ in range(11): # 10 iterations
                    input_embeddings = functional_call(self.encoder, encoder_parameters, (images,))
                    iteration_predictions.append(self.task_decoder(input_embeddings))
                    task_modality_reconstruction_loss = self.task_modality_decoder_loss(modality_reconstructions=self.task_modality_decoder(input_embeddings, images), images=images)
                    task_modality_reconstruction_loss_grads = torch.autograd.grad(task_modality_reconstruction_loss, encoder_parameters.values()) # computes the gradients of the task modality reconstruction loss with respect to the encoder parameters                        encoder_parameters = {name: param - self.lr * grad for (name, param), grad in zip(encoder_parameters.items(), task_modality_reconstruction_loss_grads)} # SGD parameter update

                    with torch.no_grad():
                        encoder_parameters = {name: param - self.lr * grad for (name, param), grad in zip(encoder_parameters.items(), task_modality_reconstruction_loss_grads)}

                    encoder_parameters = {name: parameter.detach().requires_grad_() for name, parameter in encoder_parameters.items()}

            return torch.cat(iteration_predictions, dim=1).t().unsqueeze(-1) # (num_iterations, batch_size, 1)

            # with torch.enable_grad(): # need to be able to compute gradients even during validation and testing
            #     encoder_parameters = {name: parameter.detach().clone().requires_grad_() for name, parameter in self.encoder.named_parameters()}
            #     initial_input_embeddings = functional_call(self.encoder, encoder_parameters, (images,))
            #     initial_task_prediction = self.task_decoder(initial_input_embeddings)
            #     task_modality_reconstruction_loss = self.task_modality_decoder_loss(modality_reconstructions=self.task_modality_decoder(initial_input_embeddings, images), images=images)
            #     task_modality_reconstruction_loss_grads = torch.autograd.grad(task_modality_reconstruction_loss, encoder_parameters.values()) # computes the gradients of the task modality reconstruction loss with respect to the encoder parameters
            #     adapted_encoder_parameters = {name: param - self.lr * grad for (name, param), grad in zip(encoder_parameters.items(), task_modality_reconstruction_loss_grads)}
            #     adapted_input_embeddings = functional_call(self.encoder, adapted_encoder_parameters, (images,))
            #     adapted_task_prediction = self.task_decoder(adapted_input_embeddings)

            # return initial_task_prediction, adapted_task_prediction
        elif self.adaptation_mode == 'learn_loss':
            # keep batch norm and dropout deterministic
            self.encoder.eval()
            self.task_decoder.eval()
            self.task_modality_decoder.eval()

            with torch.enable_grad(): # need to be able to compute gradients even during validation and testing
                encoder_parameters = {name: parameter.detach().clone().requires_grad_() for name, parameter in self.encoder.named_parameters()}
                initial_input_embeddings = functional_call(self.encoder, encoder_parameters, (images,))
                initial_task_prediction = self.task_decoder(initial_input_embeddings)
                surrogate_loss = self.surrogate_loss_network(self.modality_reconstruction_loss_calculator(modality_reconstructions=self.task_modality_decoder(initial_input_embeddings, images), images=images))
                surrogate_loss_grads = torch.autograd.grad(surrogate_loss.mean(), encoder_parameters.values(), retain_graph=True) # computes the gradients of the surrogate loss with respect to the encoder parameters

            with torch.no_grad():
                adapted_encoder_parameters = {name: param - self.lr * grad for (name, param), grad in zip(encoder_parameters.items(), surrogate_loss_grads)}
                adapted_input_embeddings = functional_call(self.encoder, adapted_encoder_parameters, (images,))
                adapted_task_prediction = self.task_decoder(adapted_input_embeddings)

            return initial_task_prediction, surrogate_loss, adapted_task_prediction
        elif self.adaptation_mode == 'maml':
            # keep batch norm and dropout deterministic
            self.encoder.eval()
            self.task_decoder.eval()
            self.task_modality_decoder.eval()

            with torch.enable_grad(): # need to be able to compute gradients even during validation and testing
                encoder_parameters = {name: parameter.detach().clone().requires_grad_() for name, parameter in self.encoder.named_parameters()}
                initial_input_embeddings = functional_call(self.encoder, encoder_parameters, (images,))
                initial_task_prediction = self.task_decoder(initial_input_embeddings)
                surrogate_loss = self.surrogate_loss_network(self.modality_reconstruction_loss_calculator(modality_reconstructions=self.task_modality_decoder(initial_input_embeddings, images), images=images))
                surrogate_loss_grads = torch.autograd.grad(surrogate_loss, encoder_parameters.values(), create_graph=True) # computes the gradients of the surrogate loss with respect to the encoder parameters
                adapted_encoder_parameters = {name: param - self.lr * grad for (name, param), grad in zip(encoder_parameters.items(), surrogate_loss_grads)}
                adapted_input_embeddings = functional_call(self.encoder, adapted_encoder_parameters, (images,))
                adapted_task_prediction = self.task_decoder(adapted_input_embeddings)

            return initial_task_prediction, adapted_task_prediction

class Model(LightningModule):
    def __init__(self, task, architecture, adaptation_mode, tuning_mode, pretrained, decay_factor, max_lr, weight_decay, warmup_epochs, num_train_batches, min_lr, epochs):
        super().__init__()

        self.save_hyperparameters()
        self.configure_models()
        self.configure_metrics()

        if adaptation_mode == 'task_modality_decoder':
            self.criterion = TaskModalityDecoderLoss()
        elif adaptation_mode == 'learn_loss':
            self.criterion = nn.MSELoss()
        elif task == 'species': # multi-label classification
            self.criterion = nn.BCEWithLogitsLoss()
        else: # regression
            self.criterion = nn.MSELoss()

    def configure_models(self):
        pixelwise = True if self.hparams.task == 'biomass' else False
        num_classes = 100 if self.hparams.task == 'species' else 1

        self.model = EncoderDecoder(task=self.hparams.task,
                                    architecture=self.hparams.architecture,
                                    adaptation_mode=self.hparams.adaptation_mode,
                                    tuning_mode=self.hparams.tuning_mode,
                                    pixelwise=pixelwise,
                                    num_classes=num_classes,
                                    pretrained=self.hparams.pretrained,
                                    lr=self.hparams.max_lr)

    def configure_metrics(self):
        if self.hparams.task == 'species':
            num_labels = 100
            metrics = MetricCollection({'Recall': MultilabelRecall(num_labels),
                                        'MAP': MultilabelAveragePrecision(num_labels)})
        elif self.hparams.adaptation_mode == 'tto' or self.hparams.adaptation_mode == 'maml':
            metrics = MetricCollection({'RMSE': MeanSquaredError(squared=False),
                                        'MAE': MeanAbsoluteError(),
                                        'ME': MeanError(),
                                        'R2': R2Score(),
                                        # 'adaptation improvement %': AdaptationImprovement()
                                        })
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
                if 'ResNet' in self.hparams.architecture:
                    parameters_to_unfreeze = self.model.model.model.fc.parameters() # final layer
                elif 'DINO' in self.hparams.architecture or 'MPMAE' in self.hparams.architecture:
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

                if 'ResNet' in self.hparams.architecture:
                    if num_digits == 0 or num_digits == 1:
                        layer_name = parts[0]
                    elif num_digits == 3:
                        layer_name = f'{parts[0]}.{parts[1]}'
                elif 'MPMAE' in self.hparams.architecture:
                    if num_digits == 0:
                        layer_name = parts[0]
                    elif num_digits == 1:
                        layer_name = f'{parts[0]}.{parts[1]}'
                    elif num_digits == 2:
                        layer_name = f'{parts[0]}.{parts[1]}.{parts[2]}'
                elif 'DINO' in self.hparams.architecture:
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
        assert self.trainer.num_train_batches == self.hparams.num_train_batches
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

        if self.hparams.adaptation_mode == 'tto':
            iteration_predictions = prediction
            prediction = iteration_predictions[-1]
        elif self.hparams.adaptation_mode == 'maml':
            initial_task_prediction, prediction = prediction
        elif self.hparams.adaptation_mode == 'learn_loss':
            initial_task_prediction, prediction, adapted_task_prediction = prediction # initial task prediction, surrogate loss, adapted task prediction

        batch_size = images['Sentinel2']['data'].shape[0] # number of items in batch

        if self.hparams.task == 'biomass':
            valid_mask = target != biomass_no_data_value # mask for the NaN pixels in the target

            # remove the padding added for DINOv2
            if self.hparams.architecture == 'DINOv2':
                prediction = prediction[:, :, 6:-6, 6:-6]

                if self.hparams.adaptation_mode == 'maml' or self.hparams.adaptation_mode == 'tto' or self.hparams.adaptation_mode == 'learn_loss':
                    initial_task_prediction = initial_task_prediction[:, :, 6:-6, 6:-6]

            prediction = prediction[valid_mask]
            target = target[valid_mask]

            if self.hparams.adaptation_mode == 'maml' or self.hparams.adaptation_mode == 'tto' or self.hparams.adaptation_mode == 'learn_loss':
                initial_task_prediction = initial_task_prediction[valid_mask]
            if self.hparams.adaptaiton_mode == 'learn_loss':
                adapted_task_prediction = adapted_task_prediction[valid_mask]

        if self.hparams.adaptation_mode == 'task_modality_decoder':
            target = images # task is to reconstruct the modalities

            with torch.no_grad():
                modality_reconstruction_losses = ModalityReconstructionLossCalculator()(prediction, images)

                for modality, loss in modality_reconstruction_losses.items():
                    self.log(f'{mode.capitalize()} {modality.capitalize()} reconstruction loss', loss.nanmean())
        elif self.hparams.adaptation_mode == 'learn_loss':
            task = target
            target = nn.MSELoss(reduction='none')(initial_task_prediction, target) if self.hparams.task != 'species' else nn.BCEWithLogitsLoss()(initial_task_prediction, target) # task loss

        loss = self.criterion(prediction, target) # computes the loss
        self.log(f'{get_log_name(mode, dataloader_idx)} loss', loss) # logs the loss

        # if self.hparams.adaptation_mode == 'maml':
        #     initial_loss = nn.MSELoss()(initial_task_prediction, target) if self.hparams.task != 'species' else nn.BCEWithLogitsLoss()(initial_task_prediction, target)
        #     self.log(f'{get_log_name(mode, dataloader_idx)} initial loss', initial_loss, batch_size=batch_size) # logs the loss prior to adaptation
        #     adaptation_improvement = (initial_loss - loss) / initial_loss * 100 # larger is better
        #     self.log(f'{get_log_name(mode, dataloader_idx)} loss adaptation improvement %', adaptation_improvement, batch_size=batch_size)
        # elif self.hparams.adaptation_mode == 'learn_loss':
        #     initial_loss = nn.MSELoss()(initial_task_prediction, task) if self.hparams.task != 'species' else nn.BCEWithLogitsLoss()(initial_task_prediction, task)
        #     self.log(f'{get_log_name(mode, dataloader_idx)} initial task loss', initial_loss, batch_size=batch_size) # logs the loss prior to adaptation
        #     adapted_task_loss = nn.MSELoss()(adapted_task_prediction, task) if self.hparams.task != 'species' else nn.BCEWithLogitsLoss()(adapted_task_prediction, task)
        #     adaptation_improvement = (initial_loss - adapted_task_loss) / initial_loss * 100 # larger is better
        #     self.log(f'{get_log_name(mode, dataloader_idx)} task loss adaptation improvement %', adaptation_improvement, batch_size=batch_size)

        if self.hparams.task == 'species':
            prediction = torch.sigmoid(prediction) # converts logits to probabilities
            target = target.long()

        if mode == 'train' or mode == 'val':
            metrics = getattr(self, f'{mode}_metrics')
        else: # test mode
            split = 'random_test' if dataloader_idx == 0 else 'geographic_test'
            metrics = getattr(self, f'{split}_metrics')

            if self.hparams.adaptation_mode != 'task_modality_decoder':
                if not hasattr(self, f'{split}_predictions'):
                    setattr(self, f'{split}_predictions', [])
                    setattr(self, f'{split}_targets', [])

                getattr(self, f'{split}_predictions').append(prediction.detach().cpu())
                getattr(self, f'{split}_targets').append(target.detach().cpu())

            if self.hparams.adaptation_mode == 'tto':
                if not hasattr(self, f'{split}_iteration_predictions'):
                    setattr(self, f'{split}_iteration_predictions', [])

                getattr(self, f'{split}_iteration_predictions').append(iteration_predictions.detach().cpu())

                losses = torch.stack([nn.MSELoss(reduction='none')(prediction, target) if self.hparams.task != 'species' else nn.BCEWithLogitsLoss()(prediction, target) for prediction in iteration_predictions]).squeeze(-1)

                if not hasattr(self, f'{split}_losses'):
                    setattr(self, f'{split}_losses', [])

                getattr(self, f'{split}_losses').append(losses.detach().cpu())

                if batch_idx == self.trainer.num_test_batches[dataloader_idx]-1: # if we are on the last batch
                    losses = torch.cat(getattr(self, f'{split}_losses'), dim=1).mean(dim=1)
                    initial_loss = losses[0].item()
                    final_loss = losses[-1].item()
                    adaptation_improvement = (initial_loss - final_loss) / initial_loss * 100

                    self.log(f'{split.replace("_", " ").capitalize()} adaptation improvement %', adaptation_improvement)

        if self.hparams.adaptation_mode != 'task_modality_decoder':
            for name, metric in metrics.items():
                if name == 'adaptation improvement %' and self.hparams.adaptation_mode == 'tto':
                    metric.update(initial_loss=losses[0], final_loss=losses[-1])
                    # metric.update(initial_loss=nn.MSELoss()(iteration_predictions[0], target) if self.hparams.task != 'species' else nn.BCEWithLogitsLoss()(iteration_predictions[0], target), final_loss=loss)
                else:
                    metric.update(prediction, target)

            # metrics(prediction, target) # calculates the metrics
            self.log_dict(metrics) # logs the metrics

        if batch_idx == 0: # if we are on the first batch
            images_to_log = images['Sentinel2']['data'].cpu().numpy()[:, [3,2,1]].astype(float)

            if mode == 'train' or mode == 'val':
                self._log_images(images_to_log, mode)
            else:
                self._log_images(images_to_log, split)

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
        if self.hparams.adaptation_mode != 'task_modality_decoder':
            self._plot_predictions_vs_targets('random_test')
            self._plot_predictions_vs_targets('geographic_test')

        if self.hparams.adaptation_mode == 'tto':
            self._plot_tto_loss_over_iterations('random_test')
            self._plot_tto_loss_over_iterations('geographic_test')

    def _log_images(self, images, mode):
        images = np.array([np.stack(utils.normalize(image), axis=-1) for image in images])
        num_images = min(len(images), num_logged_images)
        fig, axes = plt.subplots(1, num_images, figsize=(num_images*4, 4))
        axes = np.atleast_1d(axes)

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

        # add 1:1 line
        min_val = min(np.min(predictions), np.min(targets))
        max_val = max(np.max(predictions), np.max(targets))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 line')

        # get metrics
        metrics = getattr(self, f'{split}_metrics').compute()
        r2 = metrics[f'{split.replace("_", " ").capitalize()} R2'].item()
        rmse = metrics[f'{split.replace("_", " ").capitalize()} RMSE'].item()
        mae = metrics[f'{split.replace("_", " ").capitalize()} MAE'].item()
        me = metrics[f'{split.replace("_", " ").capitalize()} ME'].item()
        metrics_text = f'R²: {r2:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nME: {me:.4f}'
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # fit a regression line
        z = np.polyfit(x=targets, y=predictions, deg=1) # linear regression (least squares polynomial fit)
        p = np.poly1d(z) # polynomial function
        ax.plot(np.sort(targets), p(np.sort(targets)), 'b-', label=f'Fit: y={z[0]:.4f}x+{z[1]:.4f}')

        # set labels and title
        ax.set_xlabel('Target', fontsize=14)
        ax.set_ylabel('Prediction', fontsize=14)
        ax.set_title(f'{self.hparams.task.replace("_", " ").capitalize().replace("ph", "pH")} {self.hparams.architecture.capitalize().replace("Mpmae", "MPMAE").replace("_", "-").replace("mme", "MME").replace("imagenet", "ImageNet").replace("Resnet", "ResNet")} {self.hparams.tuning_mode.upper()} {split.replace("_", " ")} set', fontsize=16)
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

    def _plot_tto_loss_over_iterations(self, split):
        # iteration_predictions = torch.cat(getattr(self, f'{split}_iteration_predictions'), dim=1)
        # targets = torch.cat(getattr(self, f'{split}_targets'))
        losses = torch.cat(getattr(self, f'{split}_losses'), dim=1).mean(dim=1)
        # losses = [nn.MSELoss()(predictions, targets).item() if self.hparams.task != 'species' else nn.BCEWithLogitsLoss()(predictions, targets).item() for predictions in iteration_predictions]
        x = range(len(losses))
        plt.figure(figsize=(6, 4))
        plt.scatter(x, losses)
        plt.plot(x, losses, linestyle="-", linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Task loss')
        plt.title(f'TTO loss over iterations on {split.replace("_", " ")} set')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f'figures/{split}_tto_loss_over_iterations.png', dpi=300, bbox_inches='tight')
        plt.close()

        if wandb.run is not None:
            wandb.log({f'{split.replace("_", " ").capitalize()} TTO loss over iterations': wandb.Image(f'figures/{split}_tto_loss_over_iterations.png')})

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.hparams.adaptation_mode == 'maml':
            with torch.no_grad():
                if not hasattr(self, "_prev_surrogate_params"):
                    self._prev_surrogate_params = [p.detach().clone() for p in self.model.surrogate_loss_network.parameters()]
                else:
                    deltas = []
                    for p, prev in zip(self.model.surrogate_loss_network.parameters(), self._prev_surrogate_params):
                        deltas.append((p - prev).norm().item())
                    self.log("surrogate/param_delta_mean", float(np.mean(deltas)), on_step=True)
                    self._prev_surrogate_params = [p.detach().clone() for p in self.model.surrogate_loss_network.parameters()]

                # if not hasattr(self, "_prev_surrogate_params"):
                #     self._prev_surrogate_params = [p.detach().clone() for p in self.model.surrogate_loss_network.parameters()]
                # else:
                #     deltas = []
                #     for p, prev in zip(self.model.surrogate_loss_network.parameters(), self._prev_surrogate_params):
                #         deltas.append((p - prev).norm().item())
                #     self.log("surrogate/param_delta_mean", float(np.mean(deltas)), on_step=True)
                #     self._prev_surrogate_params = [p.detach().clone() for p in self.model.surrogate_loss_network.parameters()]


    # def on_train_batch_end(self, outputs, batch, batch_idx):
    #     if self.hparams.adaptation_mode == 'maml':
    #         print(f"\n--- Batch {batch_idx} Surrogate Network Gradients ---")
    #         for name, param in self.model.surrogate_loss_network.named_parameters():
    #             if param.grad is not None:
    #                 grad_norm = param.grad.norm().item()
    #                 grad_max = param.grad.abs().max().item()
    #                 grad_mean = param.grad.mean().item()
    #                 has_nan = param.grad.isnan().any().item()
    #                 has_inf = param.grad.isinf().any().item()

    #                 print(f"{name}: norm={grad_norm:.6f}, max={grad_max:.6f}, mean={grad_mean:.6f}, nan={has_nan}, inf={has_inf}")
    #             else:
    #                 print(f"{name}: No gradient")
