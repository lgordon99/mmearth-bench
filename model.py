# ============================================== IMPORTS ============================================== #

from convnextv2 import ConvNeXtV2, load_custom_checkpoint
from lightning.pytorch import LightningModule
from tabulate_results import get_best_run_in_sweep
from terratorch import BACKBONE_REGISTRY
from torchmetrics import Metric, MetricCollection, Recall
from torchmetrics.classification import MultilabelRecall, MultilabelAveragePrecision
from torch.func import functional_call
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score
from torchvision.models import resnet50
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
image_size = 128
architecture_properties = {'DINOv3': {'num_image_channels': 3, 'embedding_dim': 1024},
                           'MPMAE': {'num_image_channels': 12, 'embedding_dim': 320},
                           'TerraMind': {'num_image_channels': 18, 'embedding_dim': 768}}

# ============================================== FUNCTIONS ============================================== #

def get_run_id(task, architecture, adaptation_mode):
    runs = wandb.Api().runs(f'{entity}/{project}', filters={'tags': {'$in': ['beta']}})
    name = '_'.join([task, architecture, adaptation_mode])
    run = [run for run in runs if run.startswith(name)][0]
    print(run['ID'])

    return run['ID'], name

def get_state_dict(run_id, name):
    run = wandb.Api().run(f'{entity}/{project}/{run_id}')

    for artifact in run.logged_artifacts():
        if 'best' in artifact.aliases:
            artifact.download(f'/tmp/{name}')

    ckpt = torch.load(f'/tmp/{name}/model.ckpt')

    return ckpt['state_dict']

def get_model_input(architecture, images):
    if architecture == 'DINOv3':
        x = images['Sentinel2']['data'][:, [3,2,1], :, :] # extracts the RGB bands
    elif architecture == 'MPMAE':
        x = images['Sentinel2']['data']
    elif architecture == 'TerraMind':
        sentinel1_mask_vertical = images['Sentinel1']['valid_mask'][:, [0,1,4,5]] # ascending VV, VH; descending VV, VH
        asc_num_valid_pixels = sentinel1_mask_vertical[:, :2].sum(dim=(1,2,3)) # counts the number of valid pixels in the ascending VV and VH bands
        desc_num_valid_pixels = sentinel1_mask_vertical[:, 2:].sum(dim=(1,2,3)) # counts the number of valid pixels in the descending VV and VH bands
        sentinel1 = [images['Sentinel1']['data'][i, [0,1]] if asc_num_valid_pixels[i] >= desc_num_valid_pixels[i] else images['Sentinel1']['data'][i, [4,5]] for i in range(len(sentinel1_mask_vertical))]
        x = torch.cat([images['Sentinel2']['data'], torch.stack(sentinel1), images['AsterDEM']['data'][:, 0].unsqueeze(1), images['Sentinel2']['data'][:, [3,2,1]]], dim=1)

    return x

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

# ============================================== ENCODER CLASSES ============================================== #

class DINOv3Encoder(nn.Module):
    def __init__(self, pretrained):
        super().__init__()

        self.model = torch.hub.load(f'{data_dir_path}/pretrained_checkpoints/dinov3',
                                    'dinov3_vitl16',
                                    source='local',
                                    weights=f'{data_dir_path}/pretrained_checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth')

    def forward(self, images):
        embeddings = self.model.forward_features(images)['x_norm_patchtokens'].permute(0, 2, 1) # (batch_size, embedding_dim, num_patches)
        embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[1], int(np.sqrt(embeddings.shape[2])), int(np.sqrt(embeddings.shape[2]))) # (batch_size, embedding_dim, num_vertical_patches, num_horizontal_patches)

        return embeddings

class MPMAEEncoder(nn.Module):
    def __init__(self, pretrained):
        super().__init__()

        self.model = ConvNeXtV2()

        if pretrained:
            checkpoint_path = f'{data_dir_path}/pretrained_checkpoints/all_mod_atto_1M_64_uncertainty_56-8.pth' # Vishal's checkpoint
            load_custom_checkpoint(self.model, checkpoint_path) # freezing and unfreezing is done in this function

    def forward(self, images):
        embeddings = self.model(images)

        return embeddings

class TerraMindEncoder(nn.Module):
    def __init__(self, pretrained):
        super().__init__()

        self.model = BACKBONE_REGISTRY.build('terramind_v1_base', pretrained=pretrained, modalities=['S2L2A', 'S1GRD', 'DEM', 'RGB'])

    def forward(self, images):
        x = {'S2L2A': images[:, :12], # extracts the 12 bands in Sentinel-2
             'S1GRD': images[:, 12:14], # extracts the VV and VH bands from either the ascending or descending pass, whichever has more valid pixels
             'DEM': images[:, 14:15], # extracts the elevation band
             'RGB': images[:, 15:]} # extracts the RGB bands in Sentinel-2
        embeddings = self.model(x)[-1] # extracts the final block's embeddings
        embeddings = embeddings.permute(0, 2, 1) # (batch_size, embedding_dim, num_patches)
        embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[1], int(np.sqrt(embeddings.shape[2])), int(np.sqrt(embeddings.shape[2]))) # (batch_size, embedding_dim, num_vertical_patches, num_horizontal_patches)

        return embeddings

# ============================================== DECODER CLASSES ============================================== #

class LinearDecoder(nn.Module):
    def __init__(self, num_image_channels, embedding_dim, out_channels):
        super().__init__()

        self.upsample = nn.Upsample(size=image_size, mode='bilinear') # upsamples bilinearly to the image size
        self.num_image_channels = num_image_channels
        self.convolution = nn.Conv2d(in_channels=embedding_dim+num_image_channels, out_channels=out_channels, kernel_size=1) # applies a linear layer to each pixel

    def forward(self, embeddings, images):
        upsampled_embeddings = self.upsample(embeddings) # upsamples the embeddings to the image size
        concatenated = torch.cat((upsampled_embeddings, images), dim=1) # concatenates along the channel dimension
        convolved = self.convolution(concatenated) # applies the convolution to the concatenated tensor

        return convolved

class TaskDecoder(nn.Module):
    def __init__(self, architecture, pixelwise, num_classes):
        super().__init__()

        embedding_dim = architecture_properties[architecture]['embedding_dim']
        self.architecture = architecture

        if not pixelwise:
            self.decoder = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=1), # global average pooling over the spatial dimensions
                                         nn.Flatten(), # collapses the spatial dimensions
                                         nn.LayerNorm(normalized_shape=embedding_dim, eps=1e-6), # normalizes across the embedding dimension
                                         nn.Linear(in_features=embedding_dim, out_features=num_classes)) # collapses the embedding dimension to the number of classes
        else:
            self.decoder = LinearDecoder(num_image_channels=architecture_properties[architecture]['num_image_channels'],
                                         embedding_dim=embedding_dim,
                                         out_channels=1)

    def forward(self, embeddings, images=None):
        task_prediction = self.decoder(embeddings) if images is None else self.decoder(embeddings, images)

        return task_prediction

class TaskModalityDecoder(nn.Module):
    def __init__(self, num_image_channels, embedding_dim):
        super().__init__()

        self.task_modality_decoder = LinearDecoder(num_image_channels=num_image_channels,
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
                    modality_reconstruction_losses[modality] = mean.masked_fill(count == 0, float('nan')) # sets the reconstruction loss to NaN for the modality if there are no valid pixels for the modality
                else: # for biome and ecoregion
                    modality_reconstruction_losses[modality] = reconstruction_loss.masked_fill(~valid_mask, float('nan')) # sets any no data pixels to NaN
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
        in_features = 25 # 12 modalities, each with a reconstruction loss and a mask value
        self.surrogate_loss_network = nn.Linear(in_features, 1, bias=False)

        with torch.no_grad():
            self.surrogate_loss_network.weight.fill_(1.0 / in_features) # initializes all weights to 1/in_features
            # self.surrogate_loss_network.bias.zero_() # initializes bias to 0

    def forward(self, modality_reconstruction_losses, initial_task_prediction):
        stacked_modality_reconstruction_losses = torch.stack(list(modality_reconstruction_losses.values()), dim=1)
        existing_modality_mask = (~stacked_modality_reconstruction_losses.isnan()).float() # creates a mask for existing modalities (not NaN)
        filled_modality_reconstruction_losses = torch.nan_to_num(stacked_modality_reconstruction_losses, nan=0.0)
        surrogate_loss_network_input = torch.cat([filled_modality_reconstruction_losses, existing_modality_mask, initial_task_prediction], dim=1)
        surrogate_loss = self.surrogate_loss_network(surrogate_loss_network_input)

        if self.average_over_batch:
            surrogate_loss = surrogate_loss.mean()

        return surrogate_loss

class Hypernetwork(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()

        hidden_dim = 64
        num_params_to_predict = 2 * embedding_dim # gamma and beta for each embedding dimension

        self.hypernetwork = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hidden_dim, num_params_to_predict))

    def forward(self, modality_reconstruction_losses, initial_task_prediction):
        stacked_modality_reconstruction_losses = torch.stack(list(modality_reconstruction_losses.values()), dim=1)
        existing_modality_mask = (~stacked_modality_reconstruction_losses.isnan()).float() # creates a mask for existing modalities (not NaN)
        filled_modality_reconstruction_losses = torch.nan_to_num(stacked_modality_reconstruction_losses, nan=0.0)
        hypernetwork_input = torch.cat([filled_modality_reconstruction_losses, existing_modality_mask, initial_task_prediction], dim=1)
        feature_adapter_parameters = self.hypernetwork(hypernetwork_input)

        return feature_adapter_parameters

class FeatureAdapter(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_embeddings, feature_adapter_parameters):
        gamma, beta = feature_adapter_parameters.unsqueeze(-1).unsqueeze(-1).split(feature_adapter_parameters.size(1) // 2, dim=1)

        return gamma * input_embeddings + beta

class EncoderDecoder(nn.Module):
    def __init__(self, task, architecture, adaptation_mode, tuning_mode, pixelwise, num_classes, pretrained, lr):
        super().__init__()

        self.architecture = architecture
        self.adaptation_mode = adaptation_mode
        self.pixelwise = pixelwise
        self.encoder = globals()[f'{architecture}Encoder'](pretrained)
        self.lr = lr
        num_image_channels = architecture_properties[architecture]['num_image_channels']
        embedding_dim = architecture_properties[architecture]['embedding_dim']

        if adaptation_mode in ['standard', 'joint_training', 'ttt', 'maml', 'mt3', 'mt3_frozen', 'rna']:
            self.task_decoder = TaskDecoder(architecture, pixelwise, num_classes)
        elif adaptation_mode == 'multimodal':
            self.task_modality_encoder = TaskModalityEncoder()

        if adaptation_mode in ['joint_training', 'task_modality_decoder', 'ttt', 'maml', 'mt3', 'mt3_frozen', 'rna']:
            self.task_modality_decoder = TaskModalityDecoder(num_image_channels, embedding_dim)

        if adaptation_mode in ['task_modality_decoder', 'ttt', 'maml', 'mt3_frozen', 'rna']:
            state_dict = get_state_dict(*get_run_id(task, architecture, 'standard', tuning_mode))
            encoder_state_dict = {key.replace('model.encoder.', ''): value for key, value in state_dict.items() if key.startswith('model.encoder')} # filters the state_dict to only include the encoder parameters
            self.encoder.load_state_dict(encoder_state_dict)
            self.encoder.requires_grad_(False) # freezes the encoder

        if adaptation_mode in ['ttt', 'maml', 'mt3_frozen', 'rna']:
            task_decoder_state_dict = {key.replace('model.task_decoder.', ''): value for key, value in state_dict.items() if key.startswith('model.task_decoder')} # filters the state_dict to only include the decoder parameters
            self.task_decoder.load_state_dict(task_decoder_state_dict)
            self.task_decoder.requires_grad_(False) # freezes the task decoder
            task_modality_decoder_state_dict = get_state_dict(run_id='u7pagh3z', name='task_modality_decoder')
            task_modality_decoder_state_dict = {key.replace('model.task_modality_decoder.', ''): value for key, value in task_modality_decoder_state_dict.items() if key.startswith('model.task_modality_decoder')}
            self.task_modality_decoder.load_state_dict(task_modality_decoder_state_dict)

        if adaptation_mode in ['ttt', 'maml', 'rna']:
            self.task_modality_decoder.requires_grad_(False) # freezes the task modality decoder

        if adaptation_mode in ['joint_training', 'ttt', 'mt3', 'mt3_frozen']:
            self.task_modality_decoder_loss = TaskModalityDecoderLoss()
        elif adaptation_mode in ['maml', 'rna']:
            self.modality_reconstruction_loss_calculator = ModalityReconstructionLossCalculator()

        if adaptation_mode in ['maml']:
            self.surrogate_loss_network = SurrogateLossNetwork(average_over_batch=adaptation_mode=='maml')
        elif adaptation_mode == 'rna':
            self.hypernetwork = Hypernetwork(input_dim=25, embedding_dim=embedding_dim)
            self.feature_adapter = FeatureAdapter()

    def forward(self, images):
        if self.adaptation_mode == 'standard':
            input_modalities = get_model_input(architecture=self.architecture, images=images)
            input_embeddings = self.encoder(input_modalities)
            task_prediction = self.task_decoder(input_embeddings) if not self.pixelwise else self.task_decoder(input_embeddings, input_modalities)

            return task_prediction
        elif self.adaptation_mode == 'multimodal':
            input_embeddings = self.encoder(images)
            modality_embeddings = self.task_modality_encoder(images)
            concatenated_embeddings = torch.cat([input_embeddings, modality_embeddings], dim=1)
            task_prediction = self.task_decoder(concatenated_embeddings)

            return task_prediction
        elif self.adaptation_mode == 'joint_training':
            input_modalities = get_model_input(architecture=self.architecture, images=images)
            input_embeddings = self.encoder(input_modalities)
            task_prediction = self.task_decoder(input_embeddings) if not self.pixelwise else self.task_decoder(input_embeddings, input_modalities)
            modality_reconstructions = self.task_modality_decoder(input_embeddings, input_modalities)
            task_modality_reconstruction_loss = self.task_modality_decoder_loss(modality_reconstructions, images)

            return task_prediction, modality_reconstructions, task_modality_reconstruction_loss
        elif self.adaptation_mode == 'task_modality_decoder':
            self.encoder.eval() # keeps batch norm and dropout deterministic

            input_embeddings = self.encoder(images)
            modality_reconstructions = self.task_modality_decoder(input_embeddings, images)

            return modality_reconstructions
        elif self.adaptation_mode == 'ttt':
            # keep batch norm and dropout deterministic
            self.encoder.eval()
            self.task_decoder.eval()
            self.task_modality_decoder.eval()

            iteration_predictions = []

            with torch.enable_grad(): # need to be able to compute gradients even during validation and testing
                encoder_parameters = {name: parameter.detach().clone().requires_grad_() for name, parameter in self.encoder.named_parameters()}

                for _ in range(2): # iterations
                    input_embeddings = functional_call(self.encoder, encoder_parameters, (images,))
                    iteration_predictions.append(self.task_decoder(input_embeddings))
                    task_modality_reconstruction_loss = self.task_modality_decoder_loss(modality_reconstructions=self.task_modality_decoder(input_embeddings, images), images=images)
                    task_modality_reconstruction_loss_grads = torch.autograd.grad(task_modality_reconstruction_loss, encoder_parameters.values()) # computes the gradients of the task modality reconstruction loss with respect to the encoder parameters                        encoder_parameters = {name: param - self.lr * grad for (name, param), grad in zip(encoder_parameters.items(), task_modality_reconstruction_loss_grads)} # SGD parameter update

                    with torch.no_grad():
                        encoder_parameters = {name: param - self.lr * grad for (name, param), grad in zip(encoder_parameters.items(), task_modality_reconstruction_loss_grads)}

                    encoder_parameters = {name: parameter.detach().requires_grad_() for name, parameter in encoder_parameters.items()}

            return torch.cat(iteration_predictions, dim=1).t().unsqueeze(-1) # (num_iterations, batch_size, 1)
        elif self.adaptation_mode == 'maml':
            # keep batch norm and dropout deterministic
            self.encoder.eval()
            self.task_decoder.eval()
            self.task_modality_decoder.eval()

            with torch.enable_grad(): # need to be able to compute gradients even during validation and testing
                encoder_parameters = {name: parameter.detach().clone().requires_grad_() for name, parameter in self.encoder.named_parameters()}
                initial_input_embeddings = functional_call(self.encoder, encoder_parameters, (images,))
                initial_task_prediction = self.task_decoder(initial_input_embeddings)
                surrogate_loss = self.surrogate_loss_network(self.modality_reconstruction_loss_calculator(modality_reconstructions=self.task_modality_decoder(initial_input_embeddings, images), images=images), initial_task_prediction)
                surrogate_loss_grads = torch.autograd.grad(surrogate_loss, encoder_parameters.values(), create_graph=True) # computes the gradients of the surrogate loss with respect to the encoder parameters
                adapted_encoder_parameters = {name: param - self.lr * grad for (name, param), grad in zip(encoder_parameters.items(), surrogate_loss_grads)}
                adapted_input_embeddings = functional_call(self.encoder, adapted_encoder_parameters, (images,))
                adapted_task_prediction = self.task_decoder(adapted_input_embeddings)

            return torch.cat([initial_task_prediction, adapted_task_prediction], dim=1).t().unsqueeze(-1) # (num_iterations, batch_size, 1)
        elif self.adaptation_mode == 'mt3':
            with torch.enable_grad(): # need to be able to compute gradients even during validation and testing
                encoder_parameters = {name: parameter.detach().clone().requires_grad_() for name, parameter in self.encoder.named_parameters()}
                initial_input_embeddings = functional_call(self.encoder, encoder_parameters, (images,))
                initial_task_prediction = self.task_decoder(initial_input_embeddings)
                task_modality_decoder_parameters = {name: parameter.detach().clone().requires_grad_() for name, parameter in self.task_modality_decoder.named_parameters()}
                modality_reconstructions = functional_call(self.task_modality_decoder, task_modality_decoder_parameters, (initial_input_embeddings, images))
                task_modality_reconstruction_loss = self.task_modality_decoder_loss(modality_reconstructions, images)
                task_modality_reconstruction_loss_encoder_grads = torch.autograd.grad(task_modality_reconstruction_loss, encoder_parameters.values(), create_graph=True) # computes the gradients of the task modality reconstruction loss with respect to the encoder parameters                        encoder_parameters = {name: param - self.lr * grad for (name, param), grad in zip(encoder_parameters.items(), task_modality_reconstruction_loss_grads)} # SGD parameter update
                task_modality_reconstruction_loss_task_modality_decoder_grads = torch.autograd.grad(task_modality_reconstruction_loss, task_modality_decoder_parameters.values(), create_graph=True) # computes the gradients of the task modality reconstruction loss with respect to the encoder parameters                        encoder_parameters = {name: param - self.lr * grad for (name, param), grad in zip(encoder_parameters.items(), task_modality_reconstruction_loss_grads)} # SGD parameter update
                adapted_encoder_parameters = {name: param - self.lr * grad for (name, param), grad in zip(encoder_parameters.items(), task_modality_reconstruction_loss_encoder_grads)}
                adapted_task_modality_decoder_parameters = {name: param - self.lr * grad for (name, param), grad in zip(task_modality_decoder_parameters.items(), task_modality_reconstruction_loss_task_modality_decoder_grads)}
                adapted_input_embeddings = functional_call(self.encoder, adapted_encoder_parameters, (images,))
                adapted_task_prediction = self.task_decoder(adapted_input_embeddings)
                adapted_modality_reconstructions = functional_call(self.task_modality_decoder, adapted_task_modality_decoder_parameters, (adapted_input_embeddings, images))
                adapted_task_modality_reconstruction_loss = self.task_modality_decoder_loss(adapted_modality_reconstructions, images)

            return torch.cat([initial_task_prediction, adapted_task_prediction], dim=1).t().unsqueeze(-1), adapted_task_modality_reconstruction_loss # (num_iterations+1, batch_size, 1)
        elif self.adaptation_mode == 'mt3_frozen':
            # keep batch norm and dropout deterministic
            self.encoder.eval()
            self.task_decoder.eval()

            with torch.enable_grad(): # need to be able to compute gradients even during validation and testing
                encoder_parameters = {name: parameter.detach().clone().requires_grad_() for name, parameter in self.encoder.named_parameters()}
                initial_input_embeddings = functional_call(self.encoder, encoder_parameters, (images,))
                initial_task_prediction = self.task_decoder(initial_input_embeddings)
                task_modality_reconstruction_loss = self.task_modality_decoder_loss(modality_reconstructions=self.task_modality_decoder(initial_input_embeddings, images), images=images)
                task_modality_reconstruction_loss_grads = torch.autograd.grad(task_modality_reconstruction_loss, encoder_parameters.values(), create_graph=True) # computes the gradients of the task modality reconstruction loss with respect to the encoder parameters                        encoder_parameters = {name: param - self.lr * grad for (name, param), grad in zip(encoder_parameters.items(), task_modality_reconstruction_loss_grads)} # SGD parameter update
                adapted_encoder_parameters = {name: param - self.lr * grad for (name, param), grad in zip(encoder_parameters.items(), task_modality_reconstruction_loss_grads)}
                adapted_input_embeddings = functional_call(self.encoder, adapted_encoder_parameters, (images,))
                adapted_task_prediction = self.task_decoder(adapted_input_embeddings)

            return torch.cat([initial_task_prediction, adapted_task_prediction], dim=1).t().unsqueeze(-1) # (num_iterations+1, batch_size, 1)
        elif self.adaptation_mode == 'rna':
            # keep batch norm and dropout deterministic
            self.encoder.eval()
            self.task_decoder.eval()
            self.task_modality_decoder.eval()

            input_embeddings = self.encoder(images)
            initial_task_prediction = self.task_decoder(input_embeddings)
            adapted_embeddings = self.feature_adapter(input_embeddings=input_embeddings, feature_adapter_parameters=self.hypernetwork(modality_reconstruction_losses=self.modality_reconstruction_loss_calculator(modality_reconstructions=self.task_modality_decoder(input_embeddings, images), images=images), initial_task_prediction=initial_task_prediction))
            adapted_task_prediction = self.task_decoder(adapted_embeddings)

            return torch.cat([initial_task_prediction, adapted_task_prediction], dim=1).t().unsqueeze(-1) # (num_iterations+1, batch_size, 1)

class Model(LightningModule):
    def __init__(self, task, architecture, adaptation_mode, tuning_mode, pretrained, decay_factor, max_lr, weight_decay, warmup_epochs, num_train_batches, min_lr, epochs, inner_loop_lr):
        super().__init__()

        self.save_hyperparameters()
        self.configure_models()
        self.configure_metrics()

        if adaptation_mode == 'task_modality_decoder':
            self.criterion = TaskModalityDecoderLoss()
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
                                    lr=self.hparams.inner_loop_lr)

    def configure_metrics(self):
        if self.hparams.task == 'species':
            num_labels = 100
            metrics = MetricCollection({'Recall': MultilabelRecall(num_labels),
                                        'MAP': MultilabelAveragePrecision(num_labels)})
        else:
            metrics = MetricCollection({'RMSE': MeanSquaredError(squared=False),
                                        'MAE': MeanAbsoluteError(),
                                        'ME': MeanError()})

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

        if self.hparams.adaptation_mode == 'ttt' or self.hparams.adaptation_mode == 'maml' or self.hparams.adaptation_mode == 'mt3_frozen' or self.hparams.adaptation_mode == 'rna':
            iteration_predictions = prediction
            prediction = iteration_predictions[-1]
        elif self.hparams.adaptation_mode == 'joint_training':
            prediction, modality_reconstructions, task_modality_reconstruction_loss = prediction
        elif self.hparams.adaptation_mode == 'mt3':
            iteration_predictions, task_modality_reconstruction_loss = prediction
            prediction = iteration_predictions[-1]

        if self.hparams.task == 'biomass':
            valid_mask = target != biomass_no_data_value # mask for the NaN pixels in the target
            prediction = prediction[valid_mask]
            target = target[valid_mask]

            if self.hparams.adaptation_mode == 'maml' or self.hparams.adaptation_mode == 'ttt':
                initial_task_prediction = initial_task_prediction[valid_mask]

        if self.hparams.adaptation_mode == 'joint_training':
            with torch.no_grad():
                modality_reconstruction_losses = ModalityReconstructionLossCalculator()(modality_reconstructions, images)

                for modality, loss in modality_reconstruction_losses.items():
                    self.log(f'{modality.capitalize()} reconstruction loss', loss.nanmean())
        elif self.hparams.adaptation_mode == 'task_modality_decoder':
            target = images # task is to reconstruct the modalities

            with torch.no_grad():
                modality_reconstruction_losses = ModalityReconstructionLossCalculator()(prediction, images)

                for modality, loss in modality_reconstruction_losses.items():
                    self.log(f'{modality.capitalize()} reconstruction loss', loss.nanmean())

        loss = self.criterion(prediction, target) # computes the loss

        if self.hparams.adaptation_mode == 'joint_training' or self.hparams.adaptation_mode == 'mt3':
            loss += task_modality_reconstruction_loss

        self.log(f'{mode.capitalize().replace("_", " ")} loss', loss, add_dataloader_idx=False) # logs the loss

        if self.hparams.task == 'species':
            prediction = torch.sigmoid(prediction) # converts logits to probabilities
            target = target.long()

        # retrieve the number of batches in the current dataloader
        if mode == 'train':
            num_batches = self.trainer.num_training_batches
        elif mode == 'val':
            num_batches = self.trainer.num_val_batches[0]
        else:
            num_batches = self.trainer.num_test_batches[dataloader_idx]

        metrics = getattr(self, f'{mode}_metrics')

        if self.hparams.adaptation_mode != 'task_modality_decoder':
            metrics(prediction, target) # calculates the metrics
            self.log_dict(metrics, add_dataloader_idx=False) # logs the metrics

            if not hasattr(self, f'{mode}_predictions'):
                setattr(self, f'{mode}_predictions', []) # initializes to an empty list
                setattr(self, f'{mode}_targets', []) # initializes to an empty list

            getattr(self, f'{mode}_predictions').append(prediction.detach().cpu()) # appends this batch's predictions to the list
            getattr(self, f'{mode}_targets').append(target.detach().cpu()) # appends this batch's targets to the list

            if batch_idx == num_batches-1: # at the end of the epoch
                predictions = torch.cat(getattr(self, f'{mode}_predictions'))
                targets = torch.cat(getattr(self, f'{mode}_targets'))
                setattr(self, f'{mode}_r2', R2Score()(predictions, targets))
                self.log(f'{mode.capitalize().replace("_", " ")} R²', getattr(self, f'{mode}_r2'), add_dataloader_idx=False) # logs the R2 score at the end of each epoch

        # log adaptation improvement over iterations
        if self.hparams.adaptation_mode == 'ttt' or self.hparams.adaptation_mode == 'maml' or self.hparams.adaptation_mode == 'mt3' or self.hparams.adaptation_mode == 'mt3_frozen' or self.hparams.adaptation_mode == 'rna':
            losses = torch.stack([nn.MSELoss(reduction='none')(prediction, target) if self.hparams.task != 'species' else nn.BCEWithLogitsLoss()(prediction, target) for prediction in iteration_predictions]).squeeze(-1) # (num_iterations+1, batch_size)

            if not hasattr(self, f'{mode}_losses'):
                setattr(self, f'{mode}_losses', [])

            getattr(self, f'{mode}_losses').append(losses.detach().cpu())

            if batch_idx == num_batches-1: # if we are on the last batch
                losses = torch.cat(getattr(self, f'{mode}_losses'), dim=1).mean(dim=1)
                initial_loss = losses[0].item()
                final_loss = losses[-1].item()
                adaptation_improvement = (initial_loss - final_loss) / initial_loss * 100

                self.log(f'{mode.capitalize().replace("_", " ")} adaptation improvement %', adaptation_improvement, add_dataloader_idx=False)

                if 'test' in mode:
                    self._plot_loss_over_iterations(mode)

                setattr(self, f'{mode}_losses', []) # resets for the next epoch

        # log the images in the first batch
        if batch_idx == 0: # if we are on the first batch
            self._log_images(images['Sentinel2']['data'].cpu().numpy()[:, [3,2,1]].astype(float), mode)

        if mode == 'train':
            return loss

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch=batch, batch_idx=batch_idx, mode='train')

        return loss

    def validation_step(self, batch, batch_idx):
        self.general_step(batch=batch, batch_idx=batch_idx, mode='val')

    def test_step(self, batch, batch_idx, dataloader_idx):
        self.general_step(batch=batch, batch_idx=batch_idx, mode='random_test' if dataloader_idx==0 else 'geographic_test', dataloader_idx=dataloader_idx)

    def on_test_end(self):
        if self.hparams.adaptation_mode != 'task_modality_decoder':
            self._plot_predictions_vs_targets('random_test')
            self._plot_predictions_vs_targets('geographic_test')

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
        r2 = getattr(self, f'{split}_r2').item()
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

    def _plot_loss_over_iterations(self, mode):
        losses = torch.cat(getattr(self, f'{mode}_losses'), dim=1).mean(dim=1)
        x = range(len(losses))
        plt.figure(figsize=(6, 4))
        plt.scatter(x, losses)
        plt.plot(x, losses, linestyle="-", linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Task loss')
        plt.title(f'Loss over iterations on {mode.replace("_", " ")} set')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(x)
        plt.tight_layout()
        plt.savefig(f'figures/{mode}_loss_over_iterations.png', dpi=300, bbox_inches='tight')
        plt.close()

        if wandb.run is not None:
            wandb.log({f'{mode.replace("_", " ").capitalize()} loss over iterations': wandb.Image(f'figures/{mode}_loss_over_iterations.png')})
