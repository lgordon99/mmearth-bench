# ============================================== IMPORTS ============================================== #

from convnextv2 import ConvNeXtV2, load_custom_checkpoint
from datetime import date
from lightning.pytorch import LightningModule
from terratorch import BACKBONE_REGISTRY
from torch.func import functional_call
from torchgeo.models import scale_mae, ScaleMAELarge16_Weights, swin_v2_b, Swin_V2_B_Weights
from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelAveragePrecision, MultilabelRecall, MulticlassAccuracy
from torchmetrics.regression import MeanSquaredError, R2Score
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import types
import utils
import wandb

# ============================================== GLOBAL VARIABLES ============================================== #

num_logged_images = 25
fontsize = 50
pad = 10
data_dir_path = os.environ['DATA_DIR_PATH']
entity = os.environ['ENTITY']
project = os.environ['PROJECT']
task_modalities = utils.read_json(f'{data_dir_path}/task_modalities.json')
categorical_modalities = ['DynamicWorld', 'ESA_WorldCover', 'biome', 'ecoregion']
no_data_values = utils.read_json(f'{data_dir_path}/no_data_values.json')
biomass_no_data_value = -9999
image_size = 128
mini_batch_size = 16
architecture_embedding_dims = {'ConvNeXtV2A': 320,
                               'ScaleMAE': 1024,
                               'DINOv3Web': 1024,
                               'DINOv3Sat': 1024,
                               'Satlas': 1024,
                               'MPMAE': 320,
                               'TerraMind': 768,
                               'CopernicusFM': 768,
                               'TerraMindS2': 768,
                               'CopernicusFMS2': 768}
sys.path.append(f'{data_dir_path}/pretrained_checkpoints/Copernicus-FM/Copernicus-FM/src')
from model_vit import vit_base_patch16

# ============================================== ENCODER CLASSES ============================================== #

class ConvNeXtV2AEncoder(nn.Module):
    def __init__(self, *_):
        super().__init__()

        self.model = timm.create_model('convnextv2_atto.fcmae', in_chans=3, num_classes=0, global_pool='', pretrained=False)

        for module in self.model.modules():
            if (dwconv := getattr(module, 'conv_dw', None)) is not None:
                dwconv.forward = types.MethodType((lambda self, x, *args, **kwargs: type(self).forward(self, x.contiguous(), *args, **kwargs)), dwconv)

    def forward(self, images):
        embeddings = self.model(images['RGB']) # extracts the final block's embeddings

        return embeddings

class ScaleMAEEncoder(nn.Module):
    def __init__(self, *_):
        super().__init__()

        self.model = scale_mae.scalemae_large_patch16(weights=ScaleMAELarge16_Weights.FMOW_RGB, res=10, img_size=image_size)

        # remove unused parameters
        if hasattr(self.model, 'head'):
            del self.model.head

        if hasattr(self.model, 'pos_embed'):
            del self.model.pos_embed

    def forward(self, images):
        embeddings = self.model.forward_features(images['Sentinel2'])[:, 1:].permute(0, 2, 1) # (batch_size, embedding_dim, num_patches)
        num_patches_per_dimension = int(np.sqrt(embeddings.shape[2]))
        embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[1], num_patches_per_dimension, num_patches_per_dimension) # (batch_size, embedding_dim, num_vertical_patches, num_horizontal_patches)

        return embeddings

class _DINOv3BaseEncoder(nn.Module):
    def __init__(self, checkpoint_name):
        super().__init__()

        self.model = torch.hub.load(f'{data_dir_path}/pretrained_checkpoints/dinov3',
                                    'dinov3_vitl16',
                                    source='local',
                                    weights=f'{data_dir_path}/pretrained_checkpoints/{checkpoint_name}.pth')

        # remove unused layer
        if hasattr(self.model, 'local_cls_norm'):
            del self.model.local_cls_norm

    def forward(self, images):
        embeddings = self.model.forward_features(images['Sentinel2'])['x_norm_patchtokens'].permute(0, 2, 1) # (batch_size, embedding_dim, num_patches)
        num_patches_per_dimension = int(np.sqrt(embeddings.shape[2]))
        embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[1], num_patches_per_dimension, num_patches_per_dimension) # (batch_size, embedding_dim, num_vertical_patches, num_horizontal_patches)

        return embeddings

class DINOv3WebEncoder(_DINOv3BaseEncoder):
    def __init__(self, *_):
        super().__init__(checkpoint_name='dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd')

class DINOv3SatEncoder(_DINOv3BaseEncoder):
    def __init__(self, *_):
        super().__init__(checkpoint_name='dinov3_vitl16_pretrain_sat493m-eadcf0ff')

class SatlasEncoder(nn.Module):
    def __init__(self, *_):
        super().__init__()

        self.model = swin_v2_b(weights=Swin_V2_B_Weights.SENTINEL2_SI_MS_SATLAS)

        # remove unused parameters
        if hasattr(self.model, 'head'):
            del self.model.head

    def forward(self, images):
        embeddings = self.model.permute(self.model.norm(self.model.features(images['Sentinel2']))) # (batch_size, embedding_dim, num_vertical_patches, num_horizontal_patches)

        return embeddings

class MPMAEEncoder(nn.Module):
    def __init__(self, pretrained, adaptation_mode):
        super().__init__()

        self.model = ConvNeXtV2(grn_mode='simplified' if ('MT3' in adaptation_mode or 'sln' in adaptation_mode) else 'original')

        if pretrained:
            checkpoint_path = f'{data_dir_path}/pretrained_checkpoints/all_mod_atto_1M_64_uncertainty_56-8.pth' # Vishal's checkpoint
            load_custom_checkpoint(self.model, checkpoint_path) # freezing and unfreezing is done in this function

    def forward(self, images):
        embeddings = self.model(images['Sentinel2']) # (batch_size, embedding_dim, vertical embedding size, horiziontal embedding size)

        return embeddings

class _TerraMindBaseEncoder(nn.Module):
    def __init__(self, modalities, pretrained):
        super().__init__()

        self.modalities = modalities
        self.model = BACKBONE_REGISTRY.build('terramind_v1_base', pretrained=pretrained, modalities=modalities)

    def forward(self, images):
        if self.modalities == ['S2L2A']:
            x = {'S2L2A': images['Sentinel2']}
        else:
            x = {'S2L2A': images['Sentinel2'], 'S1GRD': images['Sentinel1'], 'DEM': images['AsterDEM'], 'RGB': images['RGB']}

        embeddings = self.model(x)[-1] # extracts the final block's embeddings
        embeddings = embeddings.permute(0, 2, 1) # (batch_size, embedding_dim, num_patches)
        embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[1], int(np.sqrt(embeddings.shape[2])), int(np.sqrt(embeddings.shape[2]))) # (batch_size, embedding_dim, num_vertical_patches, num_horizontal_patches)

        return embeddings

class TerraMindEncoder(_TerraMindBaseEncoder):
    def __init__(self, pretrained, _):
        super().__init__(modalities=['S2L2A', 'S1GRD', 'DEM', 'RGB'], pretrained=pretrained)

class TerraMindS2Encoder(_TerraMindBaseEncoder):
    def __init__(self, pretrained, _):
        super().__init__(modalities=['S2L2A'], pretrained=pretrained)

class _CopernicusFMBaseEncoder(nn.Module):
    def __init__(self, modalities):
        super().__init__()

        self.modalities = modalities
        self.model = vit_base_patch16(global_pool=False, return_intermediate=True, intermediate_indices=[11]) # 11 corresponds to the last block
        state_dict = torch.load(f'{data_dir_path}/pretrained_checkpoints/CopernicusFM_ViT_base_varlang_e100.pth')
        state_dict['norm.weight'] = self.model.norm.weight.clone()
        state_dict['norm.bias'] = self.model.norm.bias.clone()
        self.model.load_state_dict(state_dict)

        # remove unused layer
        if hasattr(self.model, 'coord_token') and 'longitude' in modalities:
            del self.model.coord_token

        if hasattr(self.model, 'scale_token'):
            del self.model.scale_token

        if hasattr(self.model, 'time_token') and 'time' in modalities:
            del self.model.time_token

        if hasattr(self.model, 'norm'):
            del self.model.norm

        self.patch_area = 0.0256 # ( ( 16 pixels per patch x 10 m per pixel / 1000 m per km ) ** 2 ) km^2
        self.patch_size = 16 # pixels per patch
        self.sentinel2_wavelengths = [440, 490, 560, 665, 705, 740, 783, 842, 860, 940, 1610, 2190]
        self.sentinel2_bandwidths = [20, 65, 35, 30, 15, 15, 20, 115, 20, 20, 90, 180]
        self.sentinel1_wavelengths = [50000000, 50000000]
        self.sentinel1_bandwidths = [1e9, 1e9]
        self.dem_time = (date(2015, 1, 1) - date(1970, 1, 1)).days # pretraining used that date
        self.register_buffer('dem_language_embedding', torch.load(f'{data_dir_path}/pretrained_checkpoints/Copernicus-FM/var_embed_llama3.2_1B.pt')['Copernicus Digital Elevation Model'], persistent=False) # does not get stored in checkpoint

    def forward(self, images):
        sentinel2 = images['Sentinel2']

        if self.modalities == ['Sentinel2']:
            batch_size = len(sentinel2)
            sentinel_1_2_metadata = torch.stack([torch.full((batch_size, 1), float('nan')), torch.full((batch_size, 1), float('nan')), torch.full((batch_size, 1), float('nan')), torch.full((batch_size, 1), self.patch_area)], dim=1).squeeze(-1).to(sentinel2.device)
        else:
            longitude = images['longitude']
            latitude = images['latitude']
            sentinel_1_2_metadata = torch.stack([longitude, latitude, images['time'], torch.full_like(longitude, self.patch_area)], dim=1).squeeze(-1)
            dem_metadata = torch.stack([longitude, latitude, torch.full_like(longitude, self.dem_time), torch.full_like(longitude, self.patch_area)], dim=1).squeeze(-1)

        sentinel2_embeddings = self.model.forward_features(sentinel2, sentinel_1_2_metadata, self.sentinel2_wavelengths, self.sentinel2_bandwidths, language_embed=None, input_mode='spectral', kernel_size=self.patch_size)[1][0]

        if self.modalities == ['Sentinel2']:
            embeddings = sentinel2_embeddings
        else:
            sentinel1_embeddings = self.model.forward_features(images['Sentinel1'], sentinel_1_2_metadata, self.sentinel1_wavelengths, self.sentinel1_bandwidths, language_embed=None, input_mode='spectral', kernel_size=self.patch_size)[1][0]
            dem_embeddings = self.model.forward_features(images['AsterDEM'], dem_metadata, None, None, language_embed=self.dem_language_embedding, input_mode='variable', kernel_size=self.patch_size)[1][0]
            embeddings = torch.stack([sentinel2_embeddings, sentinel1_embeddings, dem_embeddings], dim=0).mean(dim=0) # averages the embeddings of the different modalities

        return embeddings

class CopernicusFMEncoder(_CopernicusFMBaseEncoder):
    def __init__(self, *_):
        super().__init__(modalities=['Sentinel2', 'Sentinel1', 'AsterDEM', 'longitude', 'latitude', 'time'])

class CopernicusFMS2Encoder(_CopernicusFMBaseEncoder):
    def __init__(self, *_):
        super().__init__(modalities=['Sentinel2'])

# ============================================== DECODER CLASSES ============================================== #

class LinearDecoder(nn.Module):
    def __init__(self, embedding_dim, out_channels):
        super().__init__()

        self.upsample = nn.Upsample(size=image_size, mode='bilinear') # upsamples bilinearly to the image size
        self.convolution = nn.Conv2d(in_channels=embedding_dim, out_channels=out_channels, kernel_size=1) # applies a linear layer to each pixel

    def forward(self, embeddings):
        upsampled_embeddings = self.upsample(embeddings) # upsamples the embeddings to the image size
        convolved = self.convolution(upsampled_embeddings) # applies the convolution

        return convolved

class TaskDecoder(nn.Module):
    def __init__(self, architecture, adaptation_mode, pixelwise, num_classes):
        super().__init__()

        embedding_dim = architecture_embedding_dims[architecture]

        if adaptation_mode in ['multimodal', 'multimodal_joint_training', 'ttt-mjt', 'multimodal_MT3', 'multimodal_sln', 'sln_encode']:
            embedding_dim += 320 # 320 is the embedding dimension of the task modality encoder

        if not pixelwise:
            self.decoder = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=1), # global average pooling over the spatial dimensions
                                         nn.Flatten(), # collapses the spatial dimensions
                                         nn.LayerNorm(normalized_shape=embedding_dim, eps=1e-6), # normalizes across the embedding dimension
                                         nn.Linear(in_features=embedding_dim, out_features=num_classes)) # collapses the embedding dimension to the number of classes
        else:
            self.decoder = LinearDecoder(embedding_dim=embedding_dim,
                                         out_channels=1)

    def forward(self, embeddings):
        task_prediction = self.decoder(embeddings)

        return task_prediction

class TaskModalityDecoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.task_modality_decoder = LinearDecoder(embedding_dim=embedding_dim,
                                                   out_channels=922) # 922 is the total number of bands in the task modalities excluding the NaN bands in the categorical modalities
        self.modality_band_indices = {'Sentinel2': [0, 12],
                                      'Sentinel1': [12, 20],
                                      'AsterDEM': [20, 22],
                                      'ETH_GCH': [22, 24],
                                      'DynamicWorld': [24, 33],
                                      'ESA_WorldCover': [33, 44],
                                      'precipitation': [44, 47],
                                      'temperature': [47, 56],
                                      'geolocation_encoding': [56, 60],
                                      'month_encoding': [60, 62],
                                      'biome': [62, 76],
                                      'ecoregion': [76, 922]}
        self.tile_level_modality_names = ['precipitation', 'temperature', 'geolocation_encoding', 'month_encoding', 'biome', 'ecoregion']

    def forward(self, embeddings):
        modality_reconstructions = self.task_modality_decoder(embeddings)
        modality_reconstructions = {modality: modality_reconstructions[:, indices[0]:indices[1]] for modality, indices in self.modality_band_indices.items()}
        modality_reconstructions = {modality: reconstruction.mean(dim=(2,3)) if modality in self.tile_level_modality_names else reconstruction for modality, reconstruction in modality_reconstructions.items()} # collapses the spatial dimensions for the tile-level modalities

        return modality_reconstructions

# ============================================== LOSS CLASSES ============================================== #

class ModalityReconstructionLossCalculator(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss_functions = {modality: nn.CrossEntropyLoss(ignore_index=no_data_values[modality], reduction='none') if modality in categorical_modalities else nn.MSELoss(reduction='none') for modality in task_modalities}

    def forward(self, modality_reconstructions, task_modality_data):
        modality_reconstruction_losses = {}

        for modality, reconstruction in modality_reconstructions.items():
            target = task_modality_data[modality]['data']

            if modality in categorical_modalities:
                if len(target.shape) > 1:
                    target = target.squeeze(1) # removes the channel dimension

                target = target.long() # casts to torch.int64, the type for class indices in CrossEntropyLoss

            reconstruction_loss = self.loss_functions[modality](reconstruction, target)

            if 'valid_mask' in task_modality_data[modality].keys(): # for modalities that can have NaNs
                valid_mask = task_modality_data[modality]['valid_mask']

                if len(reconstruction_loss.shape) > 1: # for modalities with either channel or pixel dimensions
                    masked_reconstruction_loss = reconstruction_loss.masked_fill(~valid_mask, 0) # sets the no data pixels to zero
                    dims_to_reduce = tuple(range(1, len(masked_reconstruction_loss.shape))) # channel and pixel dimensions as present
                    sum_ = masked_reconstruction_loss.sum(dim=dims_to_reduce) # numerator in average
                    count = valid_mask.sum(dim=dims_to_reduce) # number of valid pixels in each image
                    mean = sum_ / count.clamp_min(1) # prevents division by 0
                    modality_reconstruction_losses[modality] = mean.masked_fill(count == 0, float('nan')) # sets the reconstruction loss to NaN for the modality if there are no valid pixels for the modality
                else: # for biome and ecoregion
                    modality_reconstruction_losses[modality] = reconstruction_loss.masked_fill(~valid_mask, float('nan')) # sets any no data pixels to NaN
            else: # geolocation encoding and month encoding
                modality_reconstruction_losses[modality] = torch.mean(reconstruction_loss, dim=1) # computes the mean loss per image across the channel dimension

        return modality_reconstruction_losses

class TaskModalityDecoderLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.modality_reconstruction_loss_calculator = ModalityReconstructionLossCalculator()

    def forward(self, modality_reconstructions, task_modality_data):
        modality_reconstruction_losses = self.modality_reconstruction_loss_calculator(modality_reconstructions, task_modality_data)
        mean_loss = torch.stack(list(modality_reconstruction_losses.values()), dim=1).nanmean() # computes the mean loss across all modalities and tiles, ignoring NaNs

        return mean_loss

class ModalityReconstructionPerformanceCalculator(nn.Module):
    def __init__(self):
        super().__init__()

        self.metrics = nn.ModuleDict({modality: MulticlassAccuracy(num_classes=no_data_values[modality], ignore_index=no_data_values[modality]) if modality in categorical_modalities else R2Score() for modality in task_modalities})

    def forward(self, modality_reconstructions, task_modality_data):
        modality_reconstruction_performances = {}

        for modality, reconstruction in modality_reconstructions.items():
            device = reconstruction.device
            modality_target = task_modality_data[modality]['data'].to(device)

            if modality in categorical_modalities:
                if modality in ['DynamicWorld', 'ESA_WorldCover']:
                    modality_target = modality_target.squeeze(1).long()
            else: # continuous-valued modality
                if modality not in ['geolocation_encoding', 'month_encoding']:
                    valid_mask = task_modality_data[modality]['valid_mask']
                    reconstruction = reconstruction[valid_mask]
                    modality_target = modality_target[valid_mask]

            modality_reconstruction_performances[modality] = self.metrics[modality](reconstruction, modality_target)

        return modality_reconstruction_performances

class TaskModalityDecoderWeightedLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.modality_reconstruction_loss_calculator = ModalityReconstructionLossCalculator()
        self.modality_reconstruction_performance_calculator = ModalityReconstructionPerformanceCalculator()

    def forward(self, modality_reconstructions, task_modality_data):
        modality_reconstruction_losses = self.modality_reconstruction_loss_calculator(modality_reconstructions, task_modality_data)
        modality_reconstruction_performances = self.modality_reconstruction_performance_calculator(modality_reconstructions, task_modality_data)
        weights = {}

        for modality, performance in modality_reconstruction_performances.items():
            # print(f'{modality}: performance = {performance.mean().item()}, loss = {modality_reconstruction_losses[modality].mean().item()}')
            if performance > 0:
            # if performance <= 0:
                weights[modality] = 0
            else:
                # weights[modality] = performance
                weights[modality] = 1

        weights_values = torch.tensor(list(weights.values()), device=performance.device)
        # # print(weights_values.shape)
        normalized_weights = weights_values / weights_values.sum()
        # # print("Normalized weights:", normalized_weights)
        # # print(torch.stack(list(modality_reconstruction_losses.values()), dim=1).shape)
        weighted_losses = torch.stack(list(modality_reconstruction_losses.values()), dim=1) * normalized_weights
        # # print(weighted_losses.shape)
        mean_loss = weighted_losses.nansum() # computes the weighted sum loss across all modalities and tiles, ignoring NaNs
        # mean_loss = torch.stack(list(modality_reconstruction_losses.values()), dim=1).nanmean() # computes the mean loss across all modalities and tiles, ignoring NaNs

        return mean_loss

# ============================================== MODULE CLASSES ============================================== #

class EncoderDecoder(nn.Module):
    def __init__(self, task, architecture, adaptation_mode, pixelwise, num_classes, pretrained, lr, seed):
        super().__init__()

        self.task = task
        self.architecture = architecture
        self.adaptation_mode = adaptation_mode
        self.pixelwise = pixelwise
        self.encoder = globals()[f'{architecture}Encoder'](pretrained, adaptation_mode)
        self.lr = lr
        self.seed = seed
        self.embedding_dim = architecture_embedding_dims[architecture]

        if 'TTT' in adaptation_mode or adaptation_mode in ['FT', 'LP', 'JT', 'JT_weighted_gradients', 'MT3_metabatch']:
            self.task_decoder = TaskDecoder(architecture, adaptation_mode, pixelwise, num_classes)

        if 'TTT' in adaptation_mode or adaptation_mode in ['JT', 'JT_weighted_gradients', 'TMD', 'MT3_metabatch']:
            self.task_modality_decoder = TaskModalityDecoder(self.embedding_dim)

        if adaptation_mode == 'JT':
            self.task_modality_decoder_loss = TaskModalityDecoderLoss()
        elif 'TTT' in adaptation_mode or adaptation_mode in ['JT_weighted_gradients', 'MT3_metabatch']:
            # self.task_modality_decoder_loss = TaskModalityDecoderLoss()
            self.modality_reconstruction_loss_calculator = ModalityReconstructionLossCalculator()
            # self.modality_reconstruction_performance_calculator = ModalityReconstructionPerformanceCalculator()
            # self.task_modality_decoder_loss = TaskModalityDecoderWeightedLoss()

        # load model from checkpoint
        if 'TTT' in adaptation_mode or adaptation_mode == 'TMD':
            if adaptation_mode == 'TMD':
                state_dict = self.get_state_dict('FT')
            elif 'JT-TTT' in adaptation_mode:
                state_dict = self.get_state_dict('JT')

            # load encoder weights
            encoder_state_dict = {key.removeprefix('model.encoder.'): value for key, value in state_dict.items() if key.startswith('model.encoder')} # filters the state_dict to only include the encoder parameters
            self.encoder.load_state_dict(encoder_state_dict)

        # freeze encoder
        if 'TTT' in adaptation_mode or adaptation_mode == 'LP':
            self.encoder.requires_grad_(False) # freezes the encoder

        # freeze task decoder
        if 'TTT' in adaptation_mode:
            # load task decoder weights
            # ft_state_dict = self.get_state_dict('FT')
            # task_decoder_state_dict = {key.removeprefix('model.task_decoder.'): value for key, value in ft_state_dict.items() if key.startswith('model.task_decoder')} # filters the state_dict to only include the decoder parameters
            task_decoder_state_dict = {key.removeprefix('model.task_decoder.'): value for key, value in state_dict.items() if key.startswith('model.task_decoder')} # filters the state_dict to only include the decoder parameters
            self.task_decoder.load_state_dict(task_decoder_state_dict)
            self.task_decoder.requires_grad_(False) # freezes the task decoder

            # load task modality decoder weights
            if 'FT-TTT' in adaptation_mode:
                tmd_state_dict = self.get_state_dict('TMD')
                task_modality_decoder_state_dict = {key.removeprefix('model.task_modality_decoder.'): value for key, value in tmd_state_dict.items() if key.startswith('model.task_modality_decoder')}
            else:
                task_modality_decoder_state_dict = {key.removeprefix('model.task_modality_decoder.'): value for key, value in state_dict.items() if key.startswith('model.task_modality_decoder')}

            self.task_modality_decoder.load_state_dict(task_modality_decoder_state_dict)

        # freeze task modality decoder
        if 'TTT' in adaptation_mode:
            self.task_modality_decoder.requires_grad_(False) # freezes the task modality decoder
            self.mode = 'val'
            self.val_best_num_iterations = 5

    def get_state_dict(self, adaptation_mode):
        runs = wandb.Api().runs(f'{entity}/{project}', filters={'tags': {'$in': [f'pi_{self.seed}']}}) # filters to only include runs with a certain tag
        name = '_'.join([self.task, self.architecture, adaptation_mode, str(100)]) + '_' # uses 100% train percent
        run = [run for run in runs if run.name.startswith(name)][0] # finds the run with the matching name

        # find the best model checkpoint artifact
        for artifact in run.logged_artifacts():
            if 'best' in artifact.aliases:
                artifact.download(f'/tmp/{name}')

        ckpt = torch.load(f'/tmp/{name}/model.ckpt')

        return ckpt['state_dict']

    def ttt(self, input_data, task_modality_data, num_iterations, return_all_iterations):
        self.encoder.eval()
        self.task_decoder.eval()
        self.task_modality_decoder.eval()

        if return_all_iterations:
            iteration_predictions = []

        with torch.enable_grad(): # need to be able to compute gradients even during validation and testing
            encoder_parameters = {name: parameter.detach().requires_grad_() for name, parameter in self.encoder.named_parameters()}

            for i in range(num_iterations+1): # iterations
                input_embeddings = functional_call(self.encoder, encoder_parameters, (input_data,))

                if i > 0:
                    with torch.no_grad():
                        task_prediction = self.task_decoder(input_embeddings)

                    if return_all_iterations:
                        iteration_predictions.append(task_prediction)

                    if i == num_iterations:
                        if return_all_iterations:
                            return torch.stack(iteration_predictions)
                        else:
                            return task_prediction

                    del task_prediction

                modality_reconstructions = self.task_modality_decoder(input_embeddings)
                del input_embeddings
                modality_reconstruction_losses = self.modality_reconstruction_loss_calculator(modality_reconstructions, task_modality_data)

                # modality_reconstruction_performances = self.modality_reconstruction_performance_calculator(modality_reconstructions, task_modality_data)
                # import torch.nn.functional as F

                # weights = {}

                # for modality, performance in modality_reconstruction_performances.items():
                #     if performance > 0:
                #         weights[modality] = 1 / performance

                # normalized_weights = torch.tensor(list(weights.values()), device=performance.device) / torch.tensor(list(weights.values()), device=performance.device).sum()
                # normalized_weights = F.softmax(torch.tensor(list(weights.values()), device=performance.device), dim=0)
                # normalized_weights = F.softmax(torch.tensor(list(modality_reconstruction_performances.values()), device=input_data[list(input_data.keys())[0]].device), dim=0)

                del modality_reconstructions
                gradients_per_modality = []
                # gradients_per_modality = {}

                for modality_loss in modality_reconstruction_losses.values():
                # for modality, modality_loss in modality_reconstruction_losses.items():
                    if not modality_loss.isnan().all():
                    # if not modality_loss.isnan().all() and modality_reconstruction_performances[modality] > 0:
                        gradients_per_modality.append(torch.autograd.grad(modality_loss.nanmean(), encoder_parameters.values(), retain_graph=True))
                        # gradients_per_modality[modality] = torch.autograd.grad(modality_loss.nanmean(), encoder_parameters.values(), retain_graph=True)

                del modality_reconstruction_losses
                # del modality_reconstruction_performances
                normalized_gradients_per_modality = []
                # normalized_gradients_per_modality = {}

                for modality_grads in gradients_per_modality:
                # for modality, modality_grads in gradients_per_modality.items():
                    grad_norm = torch.linalg.vector_norm(torch.cat([g.flatten() for g in modality_grads]))
                    normalized_gradients_per_modality.append(tuple(g / (grad_norm + 1e-6) for g in modality_grads))
                    # normalized_gradients_per_modality[modality] = tuple(g / (grad_norm + 1e-6) for g in modality_grads)

                # # Flatten gradients for cosine similarity
                # grads_flat = {}
                # for m, modality_grads in normalized_gradients_per_modality.items():
                #     flat_grads = [g.flatten() for g in modality_grads]
                #     grads_flat[m] = torch.cat(flat_grads)

                # modalities = list(grads_flat.keys())
                # n_modalities = len(modalities)
                # # Compute pairwise cosine similarities
                # similarities = {}
                # for i, mod_i in enumerate(modalities):
                #     sim_sum = 0
                #     for j, mod_j in enumerate(modalities):
                #         if i != j:
                #             cos_sim = F.cosine_similarity(
                #                 grads_flat[mod_i].unsqueeze(0),
                #                 grads_flat[mod_j].unsqueeze(0)
                #             )
                #             # Only count positive agreement (aligned gradients)
                #             sim_sum += max(0, cos_sim.item())

                #     # Average agreement with other modalities
                #     similarities[mod_i] = sim_sum / (n_modalities - 1)

                # # Convert to weights (softmax for smoother distribution)
                # agreement_scores = torch.tensor([similarities[m] for m in modalities], device=next(iter(grads_flat.values())).device)
                # normalized_weights = F.softmax(agreement_scores, dim=0)  # Temperature=0.1
                # # normalized_weights = agreement_scores / agreement_scores.sum()


                del gradients_per_modality
                del grad_norm
                averaged_grads = []
                # normalized_gradients_per_modality = list(normalized_gradients_per_modality.values())

                for param_idx in range(len(normalized_gradients_per_modality[0])):
                    # Stack and average gradients for this parameter
                    param_grads_across_modalities = [normalized_gradients_per_modality[m][param_idx] for m in range(len(normalized_gradients_per_modality))]
                    # averaged_grad = torch.stack(param_grads_across_modalities).mul(normalized_weights.view(-1, *([1] * (len(param_grads_across_modalities[0].shape))))).sum(dim=0)
                    averaged_grad = torch.stack(param_grads_across_modalities).mean(dim=0) # averages element-wise across modalities
                    averaged_grads.append(averaged_grad)

                del normalized_gradients_per_modality
                task_modality_reconstruction_loss_grads = tuple(averaged_grads)
                del averaged_grads
                # task_modality_reconstruction_loss = self.task_modality_decoder_loss(modality_reconstructions=self.task_modality_decoder(input_embeddings), task_modality_data=task_modality_data)
                # del input_embeddings
                # task_modality_reconstruction_loss_grads = torch.autograd.grad(task_modality_reconstruction_loss, encoder_parameters.values()) # computes the gradients of the task modality reconstruction loss with respect to the encoder parameters
                # del task_modality_reconstruction_loss
                # grad_norm = torch.linalg.vector_norm(torch.stack([g.detach().float().norm() for g in task_modality_reconstruction_loss_grads]))
                # max_norm = 1
                # scale = (max_norm / (grad_norm + 1e-6)).clamp(max=1.0)
                # scale = 1

                # unused_params = []
                # for name, param in encoder_parameters.items():
                #     grad = torch.autograd.grad(task_modality_reconstruction_loss, param, retain_graph=True, allow_unused=True)[0]
                #     if grad is None:
                #         unused_params.append(name)

                # if unused_params:
                #     print(f"Found {len(unused_params)} unused parameters:")
                #     for name in unused_params:
                #         print(f"  - {name}")
                # exit()
                with torch.no_grad():
                    # encoder_parameters = {name: (parameter - self.lr * scale * grad).detach().requires_grad_() for (name, parameter), grad in zip(encoder_parameters.items(), task_modality_reconstruction_loss_grads)}
                    encoder_parameters = {name: (parameter - self.lr * grad).detach().requires_grad_() for (name, parameter), grad in zip(encoder_parameters.items(), task_modality_reconstruction_loss_grads)}

                # del task_modality_reconstruction_loss_grads, grad_norm, scale
                del task_modality_reconstruction_loss_grads

    def ttt_adapter(self, input_data, task_modality_data, num_iterations, return_all_iterations):
        self.encoder.eval()
        self.task_decoder.eval()
        self.task_modality_decoder.eval()
        self.adapter.eval()

        if return_all_iterations:
            iteration_predictions = []

        with torch.enable_grad(): # need to be able to compute gradients even during validation and testing
            adapter_parameters = {name: parameter.detach().requires_grad_() for name, parameter in self.adapter.named_parameters()}
            input_embeddings = self.encoder(input_data)

            for i in range(num_iterations+1): # iterations
                adapted_embeddings = functional_call(self.adapter, adapter_parameters, (input_embeddings,))

                if i > 0:
                    with torch.no_grad():
                        task_prediction = self.task_decoder(adapted_embeddings)

                    if return_all_iterations:
                        iteration_predictions.append(task_prediction)

                    if i == num_iterations:
                        if return_all_iterations:
                            return torch.stack(iteration_predictions)
                        else:
                            return task_prediction

                    del task_prediction

                task_modality_reconstruction_loss = self.task_modality_decoder_loss(modality_reconstructions=self.task_modality_decoder(adapted_embeddings), task_modality_data=task_modality_data)
                del adapted_embeddings
                task_modality_reconstruction_loss_grads = torch.autograd.grad(task_modality_reconstruction_loss, adapter_parameters.values()) # computes the gradients of the task modality reconstruction loss with respect to the encoder parameters
                del task_modality_reconstruction_loss
                grad_norm = torch.linalg.vector_norm(torch.stack([g.detach().float().norm() for g in task_modality_reconstruction_loss_grads]))
                # max_norm = 1
                # scale = (max_norm / (grad_norm + 1e-6)).clamp(max=1.0)
                scale = 1

                with torch.no_grad():
                    adapter_parameters = {name: (parameter - self.lr * scale * grad).detach().requires_grad_() for (name, parameter), grad in zip(adapter_parameters.items(), task_modality_reconstruction_loss_grads)}

                del task_modality_reconstruction_loss_grads, grad_norm, scale

    def forward(self, input_data, task_modality_data):
        if self.adaptation_mode == 'FT':
            input_embeddings = self.encoder(input_data)
            task_prediction = self.task_decoder(input_embeddings)

            return task_prediction
        elif self.adaptation_mode == 'TMD':
            self.encoder.eval()

            input_embeddings = self.encoder(input_data)
            modality_reconstructions = self.task_modality_decoder(input_embeddings)

            return modality_reconstructions
        elif self.adaptation_mode == 'LP':
            self.encoder.eval()

            input_embeddings = self.encoder(input_data)
            task_prediction = self.task_decoder(input_embeddings)

            return task_prediction
        elif self.adaptation_mode == 'JT':
            input_embeddings = self.encoder(input_data)
            task_prediction = self.task_decoder(input_embeddings)
            modality_reconstructions = self.task_modality_decoder(input_embeddings)
            task_modality_reconstruction_loss = self.task_modality_decoder_loss(modality_reconstructions, task_modality_data)

            return task_prediction, modality_reconstructions, task_modality_reconstruction_loss
        elif self.adaptation_mode == 'JT_weighted_gradients':
            input_embeddings = self.encoder(input_data)
            task_prediction = self.task_decoder(input_embeddings)
            modality_reconstructions = self.task_modality_decoder(input_embeddings)
            modality_reconstruction_losses = self.modality_reconstruction_loss_calculator(modality_reconstructions, task_modality_data)

            return task_prediction, modality_reconstructions, modality_reconstruction_losses
        # elif self.adaptation_mode in ['TTT', 'TTT-Geo', 'ttt-jt', 'TTT-UDA-SS', 'uda-ttt', 'ttt-jp']:
        # elif self.adaptation_mode in ['TTT', 'TTT-Geo']:
        # elif any(string in self.adaptation_mode for string in ['TTT', 'TTT-Geo']):
        elif 'TTT' in self.adaptation_mode:
            print(f'Running TTT for {self.val_best_num_iterations} iteration(s)')
            task_prediction = self.ttt(input_data, task_modality_data, num_iterations=self.val_best_num_iterations, return_all_iterations=self.mode=='val')

            return task_prediction

class Model(LightningModule):
    def __init__(self, task, architecture, adaptation_mode, pretrained, max_lr, weight_decay, warmup_epochs, num_train_batches, min_lr, epochs, inner_loop_lr, seed):
        super().__init__()

        self.save_hyperparameters()
        self.configure_models()
        self.configure_metrics()

        if adaptation_mode == 'TMD':
            self.criterion = TaskModalityDecoderLoss()
        elif task == 'species': # multi-label classification
            self.criterion = nn.BCEWithLogitsLoss()
        else: # regression
            self.criterion = nn.MSELoss()

        if 'TTT' in adaptation_mode:
            self.val_batches_best_num_iterations = []
        elif adaptation_mode == 'JT_weighted_gradients':
            self.automatic_optimization = False

    def configure_models(self):
        pixelwise = self.hparams.task == 'biomass'
        self.num_classes = 100 if self.hparams.task == 'species' else 1
        self.model = EncoderDecoder(task=self.hparams.task,
                                    architecture=self.hparams.architecture,
                                    adaptation_mode=self.hparams.adaptation_mode,
                                    pixelwise=pixelwise,
                                    num_classes=self.num_classes,
                                    pretrained=self.hparams.pretrained,
                                    lr=self.hparams.inner_loop_lr,
                                    seed=self.hparams.seed)

    def configure_metrics(self):
        if self.hparams.adaptation_mode == 'TMD':
            metric_collection = {}
        else:
            if self.hparams.task == 'species':
                metric_collection = {'MAP': MultilabelAveragePrecision(self.num_classes), 'Recall': MultilabelRecall(self.num_classes)}
            else:
                metric_collection = {'R2': R2Score(), 'RMSE': MeanSquaredError(squared=False)}

        if self.hparams.adaptation_mode in ['TMD', 'JT', 'JT_weighted_gradients', 'MT3_metabatch']:
            for modality in task_modalities:
                if modality in categorical_modalities:
                    metric_collection[f'{modality} accuracy'] = MulticlassAccuracy(num_classes=no_data_values[modality], ignore_index=no_data_values[modality])
                else:
                    metric_collection[f'{modality} R2'] = R2Score()

        for split in ['train', 'val', 'random_test', 'geographic_test']:
            setattr(self, f'{split}_metrics', MetricCollection(metric_collection).clone(prefix=f'{split.replace("_", " ").capitalize()} '))

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.hparams.max_lr, weight_decay=self.hparams.weight_decay)
        effective_num_train_batches = math.ceil(self.hparams.num_train_batches / self.trainer.accumulate_grad_batches) # number of batches after gradient accumulation
        warmup_steps = self.hparams.warmup_epochs * effective_num_train_batches
        warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=self.hparams.min_lr/self.hparams.max_lr, total_iters=warmup_steps)
        cooldown_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(self.hparams.epochs-self.hparams.warmup_epochs)*effective_num_train_batches, eta_min=self.hparams.min_lr)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cooldown_scheduler], milestones=[warmup_steps])

        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}

    def forward(self, input_data, task_modality_data):
        return self.model(input_data, task_modality_data)

    def general_step(self, batch, batch_idx, mode, dataloader_idx=0):
        input_data, task_modality_data, target, domain = batch # extracts the images and targets for the batch
        prediction = self(input_data, task_modality_data) # forward pass

        if self.hparams.adaptation_mode in ['JT', 'MT3_metabatch', 'multimodal_joint_training', 'joint_probing']:
            prediction, modality_reconstructions, task_modality_reconstruction_loss = prediction
        elif self.hparams.adaptation_mode == 'TMD':
            modality_reconstructions = prediction.copy()
        elif 'TTT' in self.hparams.adaptation_mode and mode == 'val':
            iteration_predictions = prediction
        elif self.hparams.adaptation_mode == 'JT_weighted_gradients':
            prediction, modality_reconstructions, modality_reconstruction_losses = prediction
            task_modality_reconstruction_loss = torch.stack(list(modality_reconstruction_losses.values()), dim=1).nanmean()

        if self.hparams.adaptation_mode != 'TMD' and self.hparams.task == 'biomass':
            valid_mask = target != biomass_no_data_value # mask for the NaN pixels in the target
            target = target[valid_mask]

            if 'TTT' in self.hparams.adaptation_mode and mode == 'val':
                iteration_predictions = torch.stack([pred[valid_mask] for pred in iteration_predictions])
            else:
                prediction = prediction[valid_mask]

        # LOSS #

        if self.hparams.adaptation_mode in ['JT', 'JT_weighted_gradients', 'UDA-SS', 'multimodal_joint_training', 'joint_probing', 'MT3', 'sln', 'multimodal_MT3', 'MT3_metabatch', 'multimodal_sln', 'TMD']:
            with torch.no_grad():
                if 'modality_reconstruction_losses' not in locals():
                    modality_reconstruction_losses = ModalityReconstructionLossCalculator()(modality_reconstructions, task_modality_data)

                for modality, loss in modality_reconstruction_losses.items():
                    mean_loss = loss.nanmean()

                    if not mean_loss.isnan():
                        self.log(f'{mode.capitalize().replace("_", " ")} {modality} reconstruction loss', mean_loss, add_dataloader_idx=False)

        if self.hparams.adaptation_mode == 'TMD':
            target = task_modality_data # task is to reconstruct the modalities

        if 'TTT' in self.hparams.adaptation_mode and mode == 'val':
            iteration_losses = np.array([self.criterion(pred, target).item() for pred in iteration_predictions])
            best_iteration_number = np.argmin(iteration_losses) + 1
            print(f'Best TTT iteration: {best_iteration_number}')
            self.val_batches_best_num_iterations.append(best_iteration_number)
            prediction = iteration_predictions[best_iteration_number-1]

        loss = self.criterion(prediction, target) # computes the loss

        if self.hparams.adaptation_mode in ['JT', 'JT_weighted_gradients', 'MT3_metabatch', 'UDA-SS', 'multimodal_joint_training', 'joint_probing', 'MT3', 'sln', 'multimodal_MT3', 'multimodal_sln']:
            self.log(f'{mode.capitalize().replace("_", " ")} task modality reconstruction loss', task_modality_reconstruction_loss, add_dataloader_idx=False) # logs the task modality reconstruction loss
            loss += task_modality_reconstruction_loss

        self.log(f'{mode.capitalize().replace("_", " ")} loss', loss, add_dataloader_idx=False) # logs the loss

        # METRICS #

        if self.hparams.task == 'species':
            prediction = torch.sigmoid(prediction) # converts logits to probabilities
            target = target.long()

        metrics = getattr(self, f'{mode}_metrics')

        for name, metric in metrics.items():
            if 'modality_reconstructions' in locals() and name.split(' ')[-2] in modality_reconstructions.keys():
                modality = name.split(' ')[-2]
                modality_target = task_modality_data[modality]['data']
                reconstruction = modality_reconstructions[modality]

                if modality in categorical_modalities:
                    if modality in ['DynamicWorld', 'ESA_WorldCover']:
                        modality_target = modality_target.squeeze(1).long()
                else: # continuous-valued modality
                    if modality not in ['geolocation_encoding', 'month_encoding']:
                        valid_mask = task_modality_data[modality]['valid_mask']
                        reconstruction = reconstruction[valid_mask]
                        modality_target = modality_target[valid_mask]

                metric.update(reconstruction, modality_target)
            else: # only updates the main task metrics, not the modality-specific ones
                metric.update(prediction, target)

        self.log_dict(metrics, on_step=False, on_epoch=True, add_dataloader_idx=False) # logs the metrics at the end of each epoch

        if self.hparams.adaptation_mode in ['TMD', 'JT', 'JT_weighted_gradients'] and self.trainer.is_last_batch:
            task_modality_reconstruction_performance = torch.stack([metric.compute() for name, metric in metrics.items() if 'accuracy' in name or 'R2' in name]).nanmean() # computes the mean performance across all modalities and tiles, ignoring NaNs
            self.log(f'{mode.capitalize().replace("_", " ")} task modality reconstruction performance', task_modality_reconstruction_performance, on_step=False, on_epoch=True, add_dataloader_idx=False) # logs the task modality reconstruction performance

        # log the images in the first batch
        if batch_idx == 0: # if we are on the first batch
            self._log_images(task_modality_data['Sentinel2']['data'].cpu().numpy()[:, [3,2,1]].astype(float), mode)

        if mode == 'train':
            return loss

    def training_step(self, batch, batch_idx):
        if self.hparams.adaptation_mode == 'JT_weighted_gradients':
            optimizer = self.optimizers()
            optimizer.zero_grad()
            input_data, task_modality_data, target, domain = batch # extracts the images and targets for the batch
            prediction = self(input_data, task_modality_data) # forward pass
            prediction, modality_reconstructions, modality_reconstruction_losses = prediction

            if self.hparams.task == 'biomass':
                valid_mask = target != biomass_no_data_value # mask for the NaN pixels in the target
                target = target[valid_mask]
                prediction = prediction[valid_mask]

            with torch.no_grad():
                for modality, loss in modality_reconstruction_losses.items():
                    mean_loss = loss.nanmean()

                    if not mean_loss.isnan():
                        self.log(f'Train {modality} reconstruction loss', mean_loss, add_dataloader_idx=False)

            task_loss = self.criterion(prediction, target) # computes the task loss
            self.log(f'Train task loss', task_loss, add_dataloader_idx=False) # logs the task loss
            weighted_grads = self.compute_weighted_gradients(task_loss, modality_reconstruction_losses)

            for param, grad in zip(self.model.parameters(), weighted_grads):
                param.grad = grad

            optimizer.step()
            self.lr_schedulers().step()
            total_loss = task_loss + torch.stack(list(modality_reconstruction_losses.values()), dim=1).nanmean()
            self.log(f'Train loss', total_loss, add_dataloader_idx=False) # logs the loss

            # log the images in the first batch
            if batch_idx == 0: # if we are on the first batch
                self._log_images(task_modality_data['Sentinel2']['data'].cpu().numpy()[:, [3,2,1]].astype(float), 'train')

            return total_loss

        loss = self.general_step(batch=batch, batch_idx=batch_idx, mode='train')

        return loss

    def validation_step(self, batch, batch_idx):
        self.general_step(batch=batch, batch_idx=batch_idx, mode='val')

    def test_step(self, batch, batch_idx, dataloader_idx):
        if 'TTT' in self.hparams.adaptation_mode:
            self.model.mode = 'test'

        self.general_step(batch=batch, batch_idx=batch_idx, mode='random_test' if dataloader_idx==0 else 'geographic_test', dataloader_idx=dataloader_idx)

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch to determine best iteration for TTT"""
        if 'TTT' in self.hparams.adaptation_mode:
            # Calculate average best iteration across all batches
            self.model.val_best_num_iterations = int(round(np.mean(self.val_batches_best_num_iterations)))
            print(f'Val best num TTT iterations: {self.model.val_best_num_iterations}')

    def compute_weighted_gradients(self, task_loss, modality_reconstruction_losses):
        valid_modality_reconstruction_losses = {modality: modality_loss for modality, modality_loss in modality_reconstruction_losses.items() if not modality_loss.isnan().all()}
        encoder_params = list(self.model.encoder.parameters())
        task_decoder_params = list(self.model.task_decoder.parameters())
        task_modality_decoder_params = list(self.model.task_modality_decoder.parameters())

        # === ENCODER: Multiple losses → normalize and weight ===
        encoder_task_grads = torch.autograd.grad(task_loss, encoder_params, retain_graph=True)
        encoder_task_grads_norm = torch.sqrt(sum(g.pow(2).sum() for g in encoder_task_grads))
        encoder_task_grads_unitnormed = [g / (encoder_task_grads_norm + 1e-8) for g in encoder_task_grads]
        encoder_reconstruction_grads_unitnormed = []

        for modality_loss in valid_modality_reconstruction_losses.values():
            encoder_modality_grads = torch.autograd.grad(modality_loss.nanmean(), encoder_params, retain_graph=True)
            encoder_modality_grads_norm = torch.sqrt(sum(g.pow(2).sum() for g in encoder_modality_grads))
            encoder_reconstruction_grads_unitnormed.append([g / (encoder_modality_grads_norm + 1e-8) for g in encoder_modality_grads])

        encoder_weighted_grads = []
        task_weight = 0.5
        modality_weight = (1 - task_weight) / len(encoder_reconstruction_grads_unitnormed)

        for i in range(len(encoder_params)):
            weighted_grad = task_weight * encoder_task_grads_unitnormed[i]

            for modality_grads in encoder_reconstruction_grads_unitnormed:
                weighted_grad = weighted_grad + modality_weight * modality_grads[i]

            encoder_weighted_grads.append(weighted_grad)

        # === TASK DECODER: Single loss → don't normalize ===
        task_decoder_grads = list(torch.autograd.grad(task_loss, task_decoder_params, retain_graph=True))

        # === TASK MODALITY DECODER: Multiple losses → normalize and average ===
        task_modality_decoder_grads_unitnormed = []

        for modality_loss in valid_modality_reconstruction_losses.values():
            modality_grads = torch.autograd.grad(modality_loss.nanmean(), task_modality_decoder_params, retain_graph=True)
            modality_grads_norm = torch.sqrt(sum(g.pow(2).sum() for g in modality_grads))
            task_modality_decoder_grads_unitnormed.append([g / (modality_grads_norm + 1e-8) for g in modality_grads])

        task_modality_decoder_weighted_grads = [torch.stack([modality_grads[i] for modality_grads in task_modality_decoder_grads_unitnormed]).mean(dim=0) for i in range(len(task_modality_decoder_params))]
        all_weighted_grads = encoder_weighted_grads + task_decoder_grads + task_modality_decoder_weighted_grads

        return all_weighted_grads

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
