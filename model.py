# ============================================== IMPORTS ============================================== #

from convnextv2 import ConvNeXtV2, load_custom_checkpoint
from datetime import date
from lightning.pytorch import LightningModule
from matplotlib.colors import ListedColormap
from terratorch import BACKBONE_REGISTRY
from torch.func import functional_call
from torchgeo.models import scale_mae, ScaleMAELarge16_Weights, swin_v2_b, Swin_V2_B_Weights
from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelAveragePrecision, MultilabelRecall, MulticlassAccuracy
from torchmetrics.regression import MeanSquaredError, R2Score
import math
import matplotlib as mpl
import matplotlib.gridspec as gridspec
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

num_logged_images = 8
fontsize = 20
data_dir_path = os.environ['DATA_DIR_PATH']

sys.path.append(f'{data_dir_path}/pretrained_checkpoints/Copernicus-FM/Copernicus-FM/src')
sys.path.append(f'{data_dir_path}/pretrained_checkpoints/galileo/src')
sys.path.append(f'{data_dir_path}/pretrained_checkpoints')

from single_file_galileo import Encoder
from model_vit import vit_base_patch16

entity = os.environ['ENTITY']
project = os.environ['PROJECT']
task_modalities = utils.read_json(f'{data_dir_path}/task_modalities.json')
continuous_modalities = ['Sentinel2', 'Sentinel1', 'ASTER_GDEM', 'ETH_GCH', 'precipitation', 'temperature', 'geolocation_encoding', 'month_encoding']
categorical_modalities = ['DynamicWorld', 'ESA_WorldCover', 'biome', 'ecoregion']
pixel_level_modalities = ['Sentinel2', 'Sentinel1', 'ASTER_GDEM', 'ETH_GCH', 'DynamicWorld', 'ESA_WorldCover']
no_data_values = utils.read_json(f'{data_dir_path}/no_data_values.json')
biomass_no_data_value = -9999
image_size = 128
dynamic_world_colors_labels = {'#419BDF': 'Water',
                               '#397D49': 'Trees',
                               '#88B053': 'Grass',
                               '#7A87C6': 'Flooded vegetation',
                               '#E49635': 'Crops',
                               '#DFC35A': 'Shrub and scrub',
                               '#C4281B': 'Built',
                               '#A59B8F': 'Bare',
                               '#B39FE1': 'Snow and ice'}
esa_worldcover_colors_labels = {'#006400': 'Tree cover',
                                '#ffbb22': 'Shrubland',
                                '#ffff4c': 'Grassland',
                                '#f096ff': 'Cropland',
                                '#fa0000': 'Built-up',
                                '#b4b4b4': 'Bare/sparse\nvegetation',
                                '#f0f0f0': 'Snow and\nice',
                                '#0064c8': 'Permanent\nwater bodies',
                                '#0096a0': 'Herbaceous\nwetland',
                                '#00cf75': 'Mangroves',
                                '#fae6a0': 'Moss and\nlichen'}
architecture_embedding_dims = {'ConvNeXtV2A': 320,
                               'ConvNeXtV2AMM': 320,
                               'ScaleMAE': 1024,
                               'DINOv3Web': 1024,
                               'DINOv3Sat': 1024,
                               'SatlasNet': 1024,
                               'MPMAE': 320,
                               'AnySat': 768,
                               'TerraMind': 768,
                               'CopernicusFM': 768,
                               'Galileo': 768,
                               'AnySatS2': 768,
                               'TerraMindS2': 768,
                               'CopernicusFMS2': 768,
                               'GalileoS2': 768}

# ============================================== FUNCTIONS ============================================== #

def create_categorical_legend(colors_labels_dict, modality_data):
    cmap = ListedColormap(colors_labels_dict.keys())
    cmap.set_bad(color='black')

    return cmap(modality_data.astype(int))

def create_continuous_legend(vmin, vmax, modality_data):
    cmap = plt.get_cmap('viridis')
    cmap.set_bad(color='black')
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    return cmap(norm(modality_data))

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

class ConvNeXtV2AMMEncoder(nn.Module):
    def __init__(self, *_):
        super().__init__()

        num_pixel_level_bands = 46
        num_tile_level_bands = 880

        self.pixel_level_modality_names = task_modalities[:6]
        self.tile_level_modality_names = task_modalities[6:]
        self.model = timm.create_model('convnextv2_atto.fcmae', in_chans=num_pixel_level_bands+num_tile_level_bands, num_classes=0, global_pool='', pretrained=False)

        for module in self.model.modules():
            if (dwconv := getattr(module, 'conv_dw', None)) is not None:
                dwconv.forward = types.MethodType((lambda self, x, *args, **kwargs: type(self).forward(self, x.contiguous(), *args, **kwargs)), dwconv)

    def forward(self, images):
        pixel_level_modalities = torch.cat([images[modality]['data'] for modality in self.pixel_level_modality_names], dim=1)
        tile_level_modalities = torch.cat([images[modality]['data'] for modality in self.tile_level_modality_names], dim=1)
        tile_level_modalities_spatial = tile_level_modalities.view(*tile_level_modalities.shape, 1, 1).expand(*tile_level_modalities.shape[:2], *pixel_level_modalities.shape[2:]) # expands the tile-level modalities to match the shape of the pixel-level modalities
        modalities = torch.cat([pixel_level_modalities, tile_level_modalities_spatial], dim=1) # combines the pixel-level and tile-level modalities
        embeddings = self.model(modalities) # extracts the final block's embeddings

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

class SatlasNetEncoder(nn.Module):
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

class _AnySatBaseEncoder(nn.Module):
    def __init__(self, modalities, pretrained):
        super().__init__()

        self.modalities = modalities
        self.model = torch.hub.load('gastruc/anysat', 'anysat', pretrained=pretrained, flash_attn=False)

        if hasattr(self.model.model, 'projector_aerial'):
            del self.model.model.projector_aerial

        if hasattr(self.model.model, 'projector_aerial-flair'):
            delattr(self.model.model, 'projector_aerial-flair')

        if hasattr(self.model.model, 'projector_alos'):
            del self.model.model.projector_alos

        if hasattr(self.model.model, 'projector_l7'):
            del self.model.model.projector_l7

        if hasattr(self.model.model, 'projector_l8'):
            del self.model.model.projector_l8

        if hasattr(self.model.model, 'projector_modis'):
            del self.model.model.projector_modis

        if hasattr(self.model.model, 'projector_naip'):
            del self.model.model.projector_naip

        if hasattr(self.model.model.projector_s1, 'pad_parameter'):
            del self.model.model.projector_s1.pad_parameter

        if hasattr(self.model.model, 'projector_s1-asc'):
            delattr(self.model.model, 'projector_s1-asc')

        if hasattr(self.model.model.projector_s2, 'pad_parameter'):
            del self.model.model.projector_s2.pad_parameter

        if hasattr(self.model.model, 'projector_spot'):
            del self.model.model.projector_spot

    def forward(self, images):
        if self.modalities == ['Sentinel2']:
            x = {'s2': images['Sentinel2'].unsqueeze(1), 's2_dates': torch.zeros([len(images['Sentinel2']), 1])}
        else:
            x = {'s2': images['Sentinel2'].unsqueeze(1), 's2_dates': images['date'], 's1': images['Sentinel1'].unsqueeze(1), 's1_dates': images['date']}

        embeddings = self.model(x, patch_size=160, output='patch').permute(0, 3, 1, 2) # (batch_size, embedding_dim, num_vertical_patches, num_horizontal_patches)

        return embeddings

class AnySatEncoder(_AnySatBaseEncoder):
    def __init__(self, pretrained, _):
        super().__init__(modalities=['Sentinel2', 'Sentinel1', 'date'], pretrained=pretrained)

class AnySatS2Encoder(_AnySatBaseEncoder):
    def __init__(self, pretrained, _):
        super().__init__(modalities=['Sentinel2'], pretrained=pretrained)

class _TerraMindBaseEncoder(nn.Module):
    def __init__(self, modalities, pretrained):
        super().__init__()

        self.modalities = modalities
        self.model = BACKBONE_REGISTRY.build('terramind_v1_base', pretrained=pretrained, modalities=modalities)

    def forward(self, images):
        if self.modalities == ['S2L2A']:
            x = {'S2L2A': images['Sentinel2']}
        else:
            x = {'S2L2A': images['Sentinel2'], 'S1GRD': images['Sentinel1'], 'DEM': images['ASTER_GDEM'], 'RGB': images['RGB']}

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

        # remove unused layers
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
            dem_embeddings = self.model.forward_features(images['ASTER_GDEM'], dem_metadata, None, None, language_embed=self.dem_language_embedding, input_mode='variable', kernel_size=self.patch_size)[1][0]
            embeddings = torch.stack([sentinel2_embeddings, sentinel1_embeddings, dem_embeddings], dim=0).mean(dim=0) # averages the embeddings of the different modalities

        return embeddings

class CopernicusFMEncoder(_CopernicusFMBaseEncoder):
    def __init__(self, *_):
        super().__init__(modalities=['Sentinel2', 'Sentinel1', 'ASTER_GDEM', 'longitude', 'latitude', 'time'])

class CopernicusFMS2Encoder(_CopernicusFMBaseEncoder):
    def __init__(self, *_):
        super().__init__(modalities=['Sentinel2'])

class _GalileoBaseEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = Encoder.load_from_folder(f'{data_dir_path}/pretrained_checkpoints', device='cpu')
        self.patch_size = 16

        # remove unused layers
        if 'WC' in self.model.space_embed: # World Cereal is not one of our modalities
            del self.model.space_embed['WC']

        for key in ['TC', 'VIIRS']: # TerraClimate and VIIRS are not among our modalities
            if key in self.model.time_embed:
                del self.model.time_embed[key]

        for key in ['LS', 'WC_static']: # LandScan is not one of our modalities
            if key in self.model.static_embed:
                del self.model.static_embed[key]

    def forward(self, images):
        (space_time_token_embeddings,
         space_token_embeddings,
         time_token_embeddings,
         static_token_embeddings,
         space_time_mask,
         space_mask,
         time_mask,
         static_mask,
         _) = self.model(s_t_x=images['space_time_input'],
                         sp_x=images['space_input'],
                         t_x=images['time_input'],
                         st_x=images['static_input'],
                         s_t_m=images['space_time_mask'],
                         sp_m=images['space_mask'],
                         t_m=images['time_mask'],
                         st_m=images['static_mask'],
                         months=images['month_input'],
                         patch_size=self.patch_size)
        embeddings = self.model.apply_mask_and_average_tokens_per_patch(space_time_token_embeddings,
                                                                        space_token_embeddings,
                                                                        time_token_embeddings,
                                                                        static_token_embeddings,
                                                                        space_time_mask,
                                                                        space_mask,
                                                                        time_mask,
                                                                        static_mask)
        embeddings = embeddings.permute(0, 2, 1) # (batch_size, embedding_dim, num_patches)
        embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[1], int(np.sqrt(embeddings.shape[2])), int(np.sqrt(embeddings.shape[2]))) # (batch_size, embedding_dim, num_vertical_patches, num_horizontal_patches)

        return embeddings

class GalileoEncoder(_GalileoBaseEncoder):
    def __init__(self, *_):
        super().__init__()

class GalileoS2Encoder(_GalileoBaseEncoder):
    def __init__(self, *_):
        super().__init__()

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
                                      'ASTER_GDEM': [20, 22],
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

        if 'TTT' in adaptation_mode or adaptation_mode in ['FT', 'LP', 'JT']:
            self.task_decoder = TaskDecoder(architecture, adaptation_mode, pixelwise, num_classes)

        if 'TTT' in adaptation_mode or adaptation_mode == 'JT':
            self.task_modality_decoder = TaskModalityDecoder(self.embedding_dim)

        if adaptation_mode == 'JT':
            self.task_modality_decoder_loss = TaskModalityDecoderLoss()
        elif 'TTT' in adaptation_mode:
            self.modality_reconstruction_loss_calculator = ModalityReconstructionLossCalculator()

        # load model from checkpoint
        if 'TTT' in adaptation_mode:
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
            task_decoder_state_dict = {key.removeprefix('model.task_decoder.'): value for key, value in state_dict.items() if key.startswith('model.task_decoder')} # filters the state_dict to only include the decoder parameters
            self.task_decoder.load_state_dict(task_decoder_state_dict)
            self.task_decoder.requires_grad_(False) # freezes the task decoder

            # load task modality decoder weights
            task_modality_decoder_state_dict = {key.removeprefix('model.task_modality_decoder.'): value for key, value in state_dict.items() if key.startswith('model.task_modality_decoder')}
            self.task_modality_decoder.load_state_dict(task_modality_decoder_state_dict)

        # freeze task modality decoder
        if 'TTT' in adaptation_mode:
            self.task_modality_decoder.requires_grad_(False) # freezes the task modality decoder
            self.mode = 'val'
            self.val_best_num_iterations = 5

    def get_state_dict(self, adaptation_mode):
        runs = wandb.Api().runs(f'{entity}/{project}', filters={'tags': {'$in': [f'chi_{self.seed}']}}) # filters to only include runs with a certain tag
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

        if return_all_iterations: # if we are saving the predictions of all iterations during validation
            iteration_predictions = []

        with torch.enable_grad(): # need to be able to compute gradients even during validation and testing
            encoder_parameters = {name: parameter.detach().requires_grad_() for name, parameter in self.encoder.named_parameters()}

            for i in range(num_iterations+1): # iterations
                input_embeddings = functional_call(self.encoder, encoder_parameters, (input_data,))

                if i == 0 and not return_all_iterations: # if it is the first iteration and we are not in validation mode
                    modality_reconstructions_JT = self.task_modality_decoder(input_embeddings)
                    task_prediction_JT = self.task_decoder(input_embeddings)
                elif i > 0 and (return_all_iterations or i == num_iterations): # if we are past the first iteration, if we are saving all iterations or it is the last iteration
                    with torch.no_grad():
                        task_prediction = self.task_decoder(input_embeddings) # performs inference

                    if return_all_iterations:
                        iteration_predictions.append(task_prediction)

                    if i == num_iterations: # if it is the last iteration
                        if return_all_iterations:
                            return None, torch.stack(iteration_predictions), None, None
                        else:
                            modality_reconstructions = self.task_modality_decoder(input_embeddings)

                            return task_prediction_JT, task_prediction, modality_reconstructions_JT, modality_reconstructions

                    del task_prediction

                modality_reconstructions = self.task_modality_decoder(input_embeddings)
                del input_embeddings
                modality_reconstruction_losses = self.modality_reconstruction_loss_calculator(modality_reconstructions, task_modality_data)
                del modality_reconstructions
                gradients_per_modality = []

                for modality_loss in modality_reconstruction_losses.values():
                    if not modality_loss.isnan().all():
                        gradients_per_modality.append(torch.autograd.grad(modality_loss.nanmean(), encoder_parameters.values(), retain_graph=True)) # computes the gradients of the modality reconstruction loss averaged across tiles with respect to the encoder parameters

                del modality_reconstruction_losses
                normalized_gradients_per_modality = []

                for modality_grads in gradients_per_modality:
                    grad_norm = torch.linalg.vector_norm(torch.cat([g.flatten() for g in modality_grads])) # computes the norm of the modality gradients
                    normalized_gradients_per_modality.append(tuple(g / (grad_norm + 1e-6) for g in modality_grads)) # normalizes the modality gradients

                del gradients_per_modality
                del grad_norm
                averaged_grads = []

                for param_idx in range(len(normalized_gradients_per_modality[0])): # iterates over the encoder parameter indices
                    param_grads_across_modalities = [normalized_gradients_per_modality[m][param_idx] for m in range(len(normalized_gradients_per_modality))] # stacks the modality gradients for the parameter
                    averaged_grad = torch.stack(param_grads_across_modalities).mean(dim=0) # averages gradients across modalities for the parameter
                    averaged_grads.append(averaged_grad)

                del normalized_gradients_per_modality
                task_modality_reconstruction_loss_grads = tuple(averaged_grads) # stacks the averaged gradients for the encoder parameters
                del averaged_grads

                with torch.no_grad():
                    encoder_parameters = {name: (parameter - self.lr * grad).detach().requires_grad_() for (name, parameter), grad in zip(encoder_parameters.items(), task_modality_reconstruction_loss_grads)} # performs the gradient descent step for the encoder parameters

                del task_modality_reconstruction_loss_grads

    def forward(self, input_data, task_modality_data):
        if self.adaptation_mode == 'FT':
            input_embeddings = self.encoder(input_data)
            task_prediction = self.task_decoder(input_embeddings)

            return task_prediction
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

        if task == 'species': # multi-label classification
            self.criterion = nn.BCEWithLogitsLoss()
        else: # regression
            self.criterion = nn.MSELoss()

        if 'TTT' in adaptation_mode:
            self.val_batches_best_num_iterations = []

            if task != 'species': # only for regression tasks
                self.random_test_predictions_JT = []
                self.random_test_predictions_TTT = []
                self.random_test_targets = []
                self.geographic_test_predictions_JT = []
                self.geographic_test_predictions_TTT = []
                self.geographic_test_targets = []

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
        if self.hparams.task == 'species':
            metric_collection = {'mAP': MultilabelAveragePrecision(self.num_classes), 'Recall': MultilabelRecall(self.num_classes)}
        else:
            metric_collection = {'R2': R2Score(), 'RMSE': MeanSquaredError(squared=False)}

        if 'TTT' in self.hparams.adaptation_mode:
            task_metric_collection = metric_collection.copy()

        if 'JT' in self.hparams.adaptation_mode:
            for modality in task_modalities:
                if modality in categorical_modalities:
                    metric_collection[f'{modality} accuracy'] = MulticlassAccuracy(num_classes=no_data_values[modality], ignore_index=no_data_values[modality])
                else:
                    metric_collection[f'{modality} R2'] = R2Score()

        for split in ['train', 'val', 'random_test', 'geographic_test']:
            metrics = task_metric_collection if 'TTT' in self.hparams.adaptation_mode and split == 'val' else metric_collection
            setattr(self, f'{split}_metrics', MetricCollection(metrics).clone(prefix=f'{split.replace("_", " ").capitalize()} '))

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.hparams.max_lr, weight_decay=self.hparams.weight_decay)
        effective_num_train_batches = math.ceil(self.hparams.num_train_batches / (self.trainer.accumulate_grad_batches * torch.cuda.device_count())) # number of batches after gradient accumulation and data parallelism
        warmup_steps = self.hparams.warmup_epochs * effective_num_train_batches
        warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=self.hparams.min_lr/self.hparams.max_lr, total_iters=warmup_steps)
        cooldown_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(self.hparams.epochs-self.hparams.warmup_epochs)*effective_num_train_batches, eta_min=self.hparams.min_lr)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cooldown_scheduler], milestones=[warmup_steps])

        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}

    def forward(self, input_data, task_modality_data):
        return self.model(input_data, task_modality_data)

    def general_step(self, batch, batch_idx, mode):
        input_data, task_modality_data, target, indices, ids = batch # extracts the images and targets for the batch
        prediction = self(input_data, task_modality_data) # forward pass

        if self.hparams.adaptation_mode == 'JT':
            prediction, modality_reconstructions, task_modality_reconstruction_loss = prediction
        elif 'TTT' in self.hparams.adaptation_mode:
            task_prediction_JT, prediction, modality_reconstructions_JT, modality_reconstructions = prediction

        if 'TTT' in self.hparams.adaptation_mode and mode == 'val':
            iteration_predictions = prediction

        if self.hparams.task == 'biomass':
            valid_mask = target != biomass_no_data_value # mask for the NaN pixels in the target
            target = target[valid_mask]

            if 'TTT' in self.hparams.adaptation_mode and mode == 'val':
                iteration_predictions = torch.stack([pred[valid_mask] for pred in iteration_predictions])
            else:
                prediction = prediction[valid_mask]

                if 'TTT' in self.hparams.adaptation_mode:
                    task_prediction_JT = task_prediction_JT[valid_mask]

        if 'TTT' in self.hparams.adaptation_mode and 'test' in mode and self.hparams.task != 'species':
            getattr(self, f'{mode}_predictions_JT').append(task_prediction_JT.detach().cpu())
            getattr(self, f'{mode}_predictions_TTT').append(prediction.detach().cpu())
            getattr(self, f'{mode}_targets').append(target.detach().cpu())

        # LOSS #

        if 'JT' in self.hparams.adaptation_mode and modality_reconstructions:
            with torch.no_grad():
                modality_reconstruction_losses = ModalityReconstructionLossCalculator()(modality_reconstructions, task_modality_data)

                for modality, loss in modality_reconstruction_losses.items():
                    mean_loss = loss.nanmean()

                    if not mean_loss.isnan():
                        self.log(f'{mode.capitalize().replace("_", " ")} {modality} reconstruction loss', mean_loss, add_dataloader_idx=False)

        if 'TTT' in self.hparams.adaptation_mode and mode == 'val':
            iteration_losses = np.array([self.criterion(pred, target).item() for pred in iteration_predictions])
            best_iteration_number = np.argmin(iteration_losses) + 1
            print(f'Best TTT iteration: {best_iteration_number}')
            self.val_batches_best_num_iterations.append(best_iteration_number)
            prediction = iteration_predictions[best_iteration_number-1]

        loss = self.criterion(prediction, target) # computes the loss

        if self.hparams.adaptation_mode == 'JT':
            self.log(f'{mode.capitalize().replace("_", " ")} task modality reconstruction loss', task_modality_reconstruction_loss, add_dataloader_idx=False) # logs the task modality reconstruction loss
            loss += task_modality_reconstruction_loss

        self.log(f'{mode.capitalize().replace("_", " ")} loss', loss, add_dataloader_idx=False) # logs the loss

        # METRICS #

        if self.hparams.task == 'species':
            prediction = torch.sigmoid(prediction) # converts logits to probabilities
            target = target.long()

        metrics = getattr(self, f'{mode}_metrics')

        for name, metric in metrics.items():
            if name.split(' ')[-2] in task_modalities: # if the metric is about a modality reconstruction
                if 'JT' in self.hparams.adaptation_mode and modality_reconstructions:
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

        if 'JT' in self.hparams.adaptation_mode and modality_reconstructions:
            continuous_task_modality_reconstruction_r2 = torch.stack([metric.compute() for name, metric in metrics.items() if any(modality in name for modality in continuous_modalities)]).nanmean() # computes the mean performance across all modalities and tiles, ignoring NaNs
            categorical_task_modality_reconstruction_accuracy = torch.stack([metric.compute() for name, metric in metrics.items() if any(modality in name for modality in categorical_modalities)]).nanmean() # computes the mean performance across all modalities and tiles, ignoring NaNs
            self.log(f'{mode.capitalize().replace("_", " ")} continuous task modality reconstruction R2', continuous_task_modality_reconstruction_r2, on_step=False, on_epoch=True, add_dataloader_idx=False) # logs the task modality reconstruction performance
            self.log(f'{mode.capitalize().replace("_", " ")} categorical task modality reconstruction accuracy', categorical_task_modality_reconstruction_accuracy, on_step=False, on_epoch=True, add_dataloader_idx=False) # logs the categorical task modality reconstruction accuracy

        # log the images in the first batch
        if batch_idx == 0: # if we are on the first batch
            stylized_mode = mode.replace('_', ' ').capitalize()
            modality_data_dict = {}
            modality_reconstruction_dict = {}
            modality_reconstruction_JT_dict = {} # to distinguish between post-JT and post-TTT reconstructions during TTT

            for modality in task_modalities:
                if modality in pixel_level_modalities:
                    modality_data = task_modality_data[modality]['data'][:num_logged_images].detach().cpu().numpy().astype(float)
                    batch_indices = indices[:num_logged_images]
                    batch_ids = ids[:num_logged_images]

                    if 'JT' in self.hparams.adaptation_mode and modality_reconstructions:
                        modality_reconstruction = modality_reconstructions[modality][:num_logged_images].detach().cpu().numpy().astype(float)

                    if 'TTT' in self.hparams.adaptation_mode and modality_reconstructions_JT:
                        modality_reconstruction_JT = modality_reconstructions_JT[modality][:num_logged_images].detach().cpu().numpy().astype(float)

                    if modality == 'Sentinel2':
                        modality_data = np.array([np.stack(utils.normalize(image), axis=-1) for image in modality_data[:, [3,2,1]]]) # selects RGB bands
                        modality_data_dict[modality] = modality_data

                        if 'JT' in self.hparams.adaptation_mode and modality_reconstructions:
                            modality_reconstruction = np.array([np.stack(utils.normalize(image), axis=-1) for image in modality_reconstruction[:, [3,2,1]]]) # selects RGB bands

                        if 'TTT' in self.hparams.adaptation_mode and modality_reconstructions_JT:
                            modality_reconstruction_JT = np.array([np.stack(utils.normalize(image), axis=-1) for image in modality_reconstruction_JT[:, [3,2,1]]]) # selects RGB bands
                    elif 'JT' in self.hparams.adaptation_mode and modality_reconstructions:
                        if modality in ['Sentinel1', 'ASTER_GDEM', 'ETH_GCH']:
                            modality_data = modality_data[:, 0]
                            modality_reconstruction = modality_reconstruction[:, 0]

                            if 'TTT' in self.hparams.adaptation_mode and modality_reconstructions_JT:
                                modality_reconstruction_JT = modality_reconstruction_JT[:, 0]

                            vmin = min(modality_data.min(), modality_reconstruction.min())
                            vmax = max(modality_data.max(), modality_reconstruction.max())

                            if 'TTT' in self.hparams.adaptation_mode and modality_reconstructions_JT:
                                vmin = min(vmin, modality_reconstruction_JT.min())
                                vmax = max(vmax, modality_reconstruction_JT.max())

                            modality_data = create_continuous_legend(vmin, vmax, modality_data)
                            modality_reconstruction = create_continuous_legend(vmin, vmax, modality_reconstruction)

                            if 'TTT' in self.hparams.adaptation_mode and modality_reconstructions_JT:
                                modality_reconstruction_JT = create_continuous_legend(vmin, vmax, modality_reconstruction_JT)
                        elif modality == 'DynamicWorld':
                            modality_data = create_categorical_legend(dynamic_world_colors_labels, modality_data.squeeze())
                            modality_reconstruction = create_categorical_legend(dynamic_world_colors_labels, np.argmax(modality_reconstruction, axis=1))

                            if 'TTT' in self.hparams.adaptation_mode and modality_reconstructions_JT:
                                modality_reconstruction_JT = create_categorical_legend(dynamic_world_colors_labels, np.argmax(modality_reconstruction_JT, axis=1))
                        elif modality == 'ESA_WorldCover':
                            modality_data = create_categorical_legend(esa_worldcover_colors_labels, modality_data.squeeze())
                            modality_reconstruction = create_categorical_legend(esa_worldcover_colors_labels, np.argmax(modality_reconstruction, axis=1))

                            if 'TTT' in self.hparams.adaptation_mode and modality_reconstructions_JT:
                                modality_reconstruction_JT = create_categorical_legend(esa_worldcover_colors_labels, np.argmax(modality_reconstruction_JT, axis=1))

                        modality_data_dict[modality] = modality_data

                    if 'JT' in self.hparams.adaptation_mode and modality_reconstructions:
                        modality_reconstruction_dict[modality] = modality_reconstruction

                    if 'TTT' in self.hparams.adaptation_mode and modality_reconstructions_JT:
                        modality_reconstruction_JT_dict[modality] = modality_reconstruction_JT

            if 'JT' not in self.hparams.adaptation_mode:
                modality_reconstruction_dict = None

            if 'TTT' not in self.hparams.adaptation_mode:
                modality_reconstruction_JT_dict = None

            self._log_images(stylized_mode, batch_indices, batch_ids, modality_data_dict, modality_reconstruction_dict, modality_reconstruction_JT_dict)

        if mode == 'train':
            return loss

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch=batch, batch_idx=batch_idx, mode='train')

        return loss

    def validation_step(self, batch, batch_idx):
        self.general_step(batch=batch, batch_idx=batch_idx, mode='val')

    def test_step(self, batch, batch_idx, dataloader_idx):
        if 'TTT' in self.hparams.adaptation_mode:
            self.model.mode = 'test'

        self.general_step(batch=batch, batch_idx=batch_idx, mode='random_test' if dataloader_idx==0 else 'geographic_test')

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch to determine best iteration for TTT"""
        if 'TTT' in self.hparams.adaptation_mode:
            self.model.val_best_num_iterations = int(round(np.mean(self.val_batches_best_num_iterations))) # calculates the average best iteration across all batches
            print(f'Val best num TTT iterations: {self.model.val_best_num_iterations}')

    def on_test_epoch_end(self):
        if getattr(self, 'random_test_predictions_TTT', None):
            for split in ['random_test', 'geographic_test']:
                predictions_JT = torch.cat(getattr(self, f'{split}_predictions_JT'))
                predictions_TTT = torch.cat(getattr(self, f'{split}_predictions_TTT'))
                targets = torch.cat(getattr(self, f'{split}_targets'))
                filename = f'predictions_targets_{split}.pt'
                torch.save({'predictions_JT': predictions_JT,
                            'predictions_TTT': predictions_TTT,
                            'targets': targets},
                            filename)
                artifact = wandb.Artifact(name=filename, type='predictions_targets')
                artifact.add_file(filename)
                wandb.log_artifact(artifact)

    def _log_images(self, name, batch_indices, batch_ids, modality_data_dict, modality_reconstruction_dict, modality_reconstruction_JT_dict):
        modalities = list(modality_data_dict.keys())
        num_images = len(modality_data_dict[modalities[0]])
        height_ratios = [1]

        for i, modality in enumerate(modalities):
            height_ratios.append(4) # ground-truth row

            if modality_reconstruction_dict:
                height_ratios.append(4) # reconstruction row

            if modality_reconstruction_JT_dict:
                height_ratios.append(4) # JT reconstruction row

            if i < len(modalities) - 1: # gap between modalities
                height_ratios.append(1)

        fig = plt.figure(figsize=(num_images * 2, sum(height_ratios) * 0.32))
        gs = gridspec.GridSpec(len(height_ratios), num_images, figure=fig, height_ratios=height_ratios, hspace=0.1, wspace=0.02)
        grid_row = 1 # starts after the initial gap
        label_positions = [] # stores gap row indices for later labeling
        row_labels = [] # stores (row_index, label_text) for Original/Reconstruction labels

        for mod_idx, modality in enumerate(modalities):
            gap_row = grid_row - 1
            label_positions.append((gap_row, modality))
            modality_data = modality_data_dict[modality]

            # Plot ground truth row
            for col in range(num_images):
                ax = fig.add_subplot(gs[grid_row, col])
                ax.imshow(modality_data[col])
                ax.axis('off')

                if mod_idx == 0: # only adds titles to first modality
                    ax.set_title(f'Idx: {batch_indices[col]}\nID: {batch_ids[col]}', fontsize=fontsize, pad=30)

            row_labels.append((grid_row, 'Original'))
            grid_row += 1

            # Plot JT reconstruction row if it exists (for TTT)
            if modality_reconstruction_JT_dict and modality in modality_reconstruction_JT_dict:
                modality_reconstruction_JT = modality_reconstruction_JT_dict[modality]

                for col in range(num_images):
                    ax = fig.add_subplot(gs[grid_row, col])
                    ax.imshow(modality_reconstruction_JT[col])
                    ax.axis('off')

                row_labels.append((grid_row, 'JT\nreconstruction'))
                grid_row += 1

            # Plot reconstruction row if it exists
            if modality_reconstruction_dict and modality in modality_reconstruction_dict:
                modality_reconstruction = modality_reconstruction_dict[modality]

                for col in range(num_images):
                    ax = fig.add_subplot(gs[grid_row, col])
                    ax.imshow(modality_reconstruction[col])
                    ax.axis('off')

                row_labels.append((grid_row, 'TTT-MMR\nreconstruction' if modality_reconstruction_JT_dict else 'Reconstruction'))
                grid_row += 1

            if mod_idx < len(modalities) - 1:
                grid_row += 1

        plt.subplots_adjust(top=0.92, bottom=0.02, left=0.08)

        # Add modality labels after layout is adjusted
        for gap_row, modality in label_positions:
            gap_row_center = (gs[gap_row, 0].get_position(fig).y0 + gs[gap_row, 0].get_position(fig).y1) / 2
            fig.text(0.5, gap_row_center, modality.replace('l2', 'l-2').replace('l1', 'l-1').replace('_', ' ').replace('cW', 'c W'), ha='center', va='center', fontsize=fontsize, transform=fig.transFigure)

        for row_idx, label_text in row_labels:
            row_center = (gs[row_idx, 0].get_position(fig).y0 + gs[row_idx, 0].get_position(fig).y1) / 2
            left_edge = gs[row_idx, 0].get_position(fig).x0
            fig.text(left_edge, row_center, label_text, ha='center', va='center', fontsize=0.5*fontsize, rotation=90, transform=fig.transFigure)

        os.makedirs('figures', exist_ok=True)
        plt.savefig(f'figures/{name}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        if wandb.run is not None:
            wandb.log({name: wandb.Image(f'figures/{name}.png')})
