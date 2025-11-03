# ============================================== IMPORTS ============================================== #

from convnextv2 import ConvNeXtV2, load_custom_checkpoint
from datetime import date
from lightning.pytorch import LightningModule
from terratorch import BACKBONE_REGISTRY
from torch.func import functional_call
from torch.nn.attention import sdpa_kernel, SDPBackend
from torchgeo.models import scale_mae, ScaleMAELarge16_Weights, swin_v2_b, Swin_V2_B_Weights
from torchmetrics import Metric, MetricCollection
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
                               'ConvNeXtV2AMultimodal': 320,
                               'ScaleMAE': 1024,
                               'DINOv3Web': 1024,
                               'DINOv3Sat': 1024,
                               'Satlas': 1024,
                               'MPMAE': 320,
                               'AnySat': 768,
                               'TerraMind': 768,
                               'CopernicusFM': 768,
                               'TerraMindS2': 768,
                               'CopernicusFMS2': 768,
                               'AnySatS2': 768}
sys.path.append(f'{data_dir_path}/pretrained_checkpoints/Copernicus-FM/Copernicus-FM/src')
from model_vit import vit_base_patch16

# ============================================== FUNCTIONS ============================================== #

def get_state_dict(task, architecture, adaptation_mode):
    runs = wandb.Api().runs(f'{entity}/{project}', filters={'tags': {'$in': ['pi']}}) # filters to only include runs with a certain tag
    name = '_'.join([task, architecture, adaptation_mode, str(100)]) + '_' # uses 100% train percent
    run = [run for run in runs if run.name.startswith(name)][0] # finds the run with the matching name

    # find the best model checkpoint artifact
    for artifact in run.logged_artifacts():
        if 'best' in artifact.aliases:
            artifact.download(f'/tmp/{name}')

    ckpt = torch.load(f'/tmp/{name}/model.ckpt')

    return ckpt['state_dict']

# ============================================== METRIC CLASSES ============================================== #

class DomainEmbeddingDistance(Metric):
    """
    Distance between mean source and mean target embeddings in feature space φ.

    Args:
        embedding_dim: dimensionality of φ(x) after pooling (D)
    """

    def __init__(self, embedding_dim):
        super().__init__()

        # running sums and counts per domain
        self.add_state('sum_source', default=torch.zeros(embedding_dim))
        self.add_state('sum_target', default=torch.zeros(embedding_dim))
        self.add_state('count_source', default=torch.tensor(0, dtype=torch.long))
        self.add_state('count_target', default=torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def update(self, embeddings, domain):
        tile_embeddings = embeddings.mean(dim=(2, 3)) # (B, D)
        source_mask = torch.tensor(['source' in d for d in domain]) # creates mask for source samples
        target_mask = torch.tensor(['target' in d for d in domain]) # creates mask for target samples

        if source_mask.any():
            self.sum_source += tile_embeddings[source_mask].sum(dim=0)
            self.count_source += source_mask.sum()

        if target_mask.any():
            self.sum_target += tile_embeddings[target_mask].sum(dim=0)
            self.count_target += target_mask.sum()

    def compute(self):
        mean_source = self.sum_source / self.count_source
        mean_target = self.sum_target / self.count_target

        return torch.linalg.vector_norm(mean_source - mean_target, ord=2) # L2 norm

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

class ConvNeXtV2AMultimodalEncoder(nn.Module):
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
        use_onehot = images['DynamicWorld']['data'].shape[1] == 1

        def get_modality_data(modality):
            if use_onehot and modality in ['DynamicWorld', 'ESA_WorldCover', 'biome', 'ecoregion']:
                return images[f'{modality}_onehot']['data']
            else:
                return images[modality]['data']

        pixel_level_modalities = torch.cat([get_modality_data(modality) for modality in self.pixel_level_modality_names], dim=1)
        tile_level_modalities = torch.cat([get_modality_data(modality) for modality in self.tile_level_modality_names], dim=1)
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
        self.dem_time = (date(2015, 1, 1) - date(1970, 1, 1)).days # pretraining was with that date
        self.register_buffer('dem_language_embedding', torch.load(f'{data_dir_path}/pretrained_checkpoints/Copernicus-FM/var_embed_llama3.2_1B.pt')['Copernicus Digital Elevation Model'], persistent=False)

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

class TaskModalityEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        num_pixel_level_bands = 46
        num_tile_level_bands = 880

        self.pixel_level_modality_names = task_modalities[:6]
        self.tile_level_modality_names = task_modalities[6:]
        self.encoder = timm.create_model('convnextv2_atto.fcmae', in_chans=num_pixel_level_bands+num_tile_level_bands, num_classes=0, global_pool='', pretrained=False)

        for module in self.encoder.modules():
            if (dwconv := getattr(module, 'conv_dw', None)) is not None:
                dwconv.forward = types.MethodType((lambda self, x, *args, **kwargs: type(self).forward(self, x.contiguous(), *args, **kwargs)), dwconv)

    def forward(self, task_modality_data):
        use_onehot = task_modality_data['DynamicWorld']['data'].shape[1] == 1

        def get_modality_data(modality):
            if use_onehot and modality in ['DynamicWorld', 'ESA_WorldCover', 'biome', 'ecoregion']:
                return task_modality_data[f'{modality}_onehot']['data']
            else:
                return task_modality_data[modality]['data']

        pixel_level_modalities = torch.cat([get_modality_data(modality) for modality in self.pixel_level_modality_names], dim=1)
        tile_level_modalities = torch.cat([get_modality_data(modality) for modality in self.tile_level_modality_names], dim=1)
        tile_level_modalities_spatial = tile_level_modalities.view(*tile_level_modalities.shape, 1, 1).expand(*tile_level_modalities.shape[:2], *pixel_level_modalities.shape[2:]) # expands the tile-level modalities to match the shape of the pixel-level modalities
        modalities = torch.cat([pixel_level_modalities, tile_level_modalities_spatial], dim=1) # combines the pixel-level and tile-level modalities
        modality_embeddings = self.encoder(modalities) # extracts the final block's embeddings

        return modality_embeddings

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

# class TaskModalityDecoder(nn.Module):
#     def __init__(self, embedding_dim):
#         super().__init__()

#         self.trunk = nn.Sequential(nn.Upsample(size=image_size, mode='bilinear'), # upsamples embeddings bilinearly to the image size
#                                    nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding=1, groups=embedding_dim), # depthwise convolution (spatial per-channel)
#                                    nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1), # pointwise convolution (channel mixing)
#                                    nn.GroupNorm(num_groups=1, num_channels=embedding_dim), # LayerNorm2D surrogate
#                                    nn.GELU(), # non-linearity
#                                    nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding=1, groups=embedding_dim), # depthwise convolution (spatial per-channel)
#                                    nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1), # pointwise convolution (channel mixing)
#                                    nn.GroupNorm(num_groups=1, num_channels=embedding_dim), # LayerNorm2D surrogate
#                                    nn.GELU(), # non-linearity
#                                    nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1)) # final pointwise convolution to produce feature maps

#         modality_info = {'Sentinel2': {'channels': 12, 'scale': 'pixel', 'type': 'continuous'},
#                          'Sentinel1': {'channels': 8, 'scale': 'pixel', 'type': 'continuous'},
#                          'AsterDEM': {'channels': 2, 'scale': 'pixel', 'type': 'continuous'},
#                          'ETH_GCH': {'channels': 2, 'scale': 'pixel', 'type': 'continuous'},
#                          'DynamicWorld': {'channels': 9, 'scale': 'pixel', 'type': 'categorical'},
#                          'ESA_WorldCover': {'channels': 11, 'scale': 'pixel', 'type': 'categorical'},
#                          'precipitation': {'channels': 3, 'scale': 'tile', 'type': 'continuous'},
#                          'temperature': {'channels': 9, 'scale': 'tile', 'type': 'continuous'},
#                          'geolocation_encoding': {'channels': 4, 'scale': 'tile', 'type': 'continuous'},
#                          'month_encoding': {'channels': 2, 'scale': 'tile', 'type': 'continuous'},
#                          'biome': {'channels': 14, 'scale': 'tile', 'type': 'categorical'},
#                          'ecoregion': {'channels': 846, 'scale': 'tile', 'type': 'categorical'}}
#         self.modality_heads = nn.ModuleDict()

#         for modality, info in modality_info.items():
#             if info['scale'] == 'pixel':
#                 self.modality_heads[modality] = nn.Conv2d(in_channels=embedding_dim, out_channels=info['channels'], kernel_size=1)
#             else:
#                 self.modality_heads[modality] = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=1),
#                                                               nn.Flatten(), # collapses the spatial dimensions
#                                                               nn.LayerNorm(embedding_dim),
#                                                               nn.Linear(embedding_dim, 256),
#                                                               nn.GELU(),
#                                                               nn.Linear(256, info['channels']))

#     def forward(self, embeddings):
#         feature_maps = self.trunk(embeddings)
#         modality_reconstructions = {modality: head(feature_maps) for modality, head in self.modality_heads.items()}

#         return modality_reconstructions

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

# class TaskModalityDecoderLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.modality_reconstruction_loss_calculator = ModalityReconstructionLossCalculator()
#         self.log_variances = nn.ParameterDict({m: nn.Parameter(torch.zeros(())) for m in task_modalities}) # one log variance per modality

#     def forward(self, modality_reconstructions, task_modality_data):
#         modality_reconstruction_losses = self.modality_reconstruction_loss_calculator(modality_reconstructions, task_modality_data)
#         total_loss = 0

#         for modality, loss in modality_reconstruction_losses.items():
#             if loss.isnan().all():
#                 continue # skips modalities that have no valid pixels in the batch

#             loss = loss.nanmean() # computes the mean loss for the modality over the batch
#             log_variance = self.log_variances[modality]
#             weight = torch.exp(-log_variance)
#             total_loss += weight * loss + log_variance

#         return total_loss

# ============================================== MODULE CLASSES ============================================== #

class SurrogateLossNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        in_features = 24 # 12 modalities, each with a reconstruction loss and a mask value
        self.surrogate_loss_network = nn.Linear(in_features=in_features, out_features=1, bias=False)

        with torch.no_grad():
            self.surrogate_loss_network.weight.fill_(1.0 / in_features) # initializes all weights to 1/in_features

    def forward(self, modality_reconstruction_losses):
        stacked_modality_reconstruction_losses = torch.stack(list(modality_reconstruction_losses.values()), dim=1)
        existing_modality_mask = (~stacked_modality_reconstruction_losses.isnan()).float() # creates a mask for existing modalities (not NaN)
        filled_modality_reconstruction_losses = torch.nan_to_num(stacked_modality_reconstruction_losses, nan=0.0)
        surrogate_loss_network_input = torch.cat([filled_modality_reconstruction_losses, existing_modality_mask], dim=1)
        surrogate_loss = self.surrogate_loss_network(surrogate_loss_network_input).mean()

        return surrogate_loss

# class SurrogateLossNetwork(nn.Module):
#     def __init__(self, in_features):
#         super().__init__()

#         # hidden_dim = 128
#         # self.surrogate_loss_network = nn.Sequential(nn.LayerNorm(in_features),
#         #                                             nn.Linear(in_features, hidden_dim),
#         #                                             nn.ReLU(),
#         #                                             nn.Linear(hidden_dim, hidden_dim // 2),
#         #                                             nn.ReLU(),
#         #                                             nn.Linear(hidden_dim // 2, 1),
#         #                                             nn.Softplus()) # ensures the output is non-negative
#         self.surrogate_loss_network = nn.Linear(in_features=in_features, out_features=1, bias=False)

#         with torch.no_grad():
#             self.surrogate_loss_network.weight.fill_(1.0 / in_features) # initializes all weights to 1/in_features

#     def forward(self, modality_reconstruction_losses):
#         stacked_modality_reconstruction_losses = torch.stack(list(modality_reconstruction_losses.values()), dim=1)
#         existing_modality_mask = (~stacked_modality_reconstruction_losses.isnan()).float() # creates a mask for existing modalities (not NaN)
#         filled_modality_reconstruction_losses = torch.nan_to_num(stacked_modality_reconstruction_losses, nan=0.0)

#         # if len(initial_task_prediction.shape) > 2:
#         #     initial_task_prediction = initial_task_prediction.mean(dim=(2,3)) # collapses the spatial dimensions for pixelwise tasks

#         # surrogate_loss_network_input = torch.cat([pooled_embeddings, initial_task_prediction], dim=1)
#         # surrogate_loss = self.surrogate_loss_network(surrogate_loss_network_input).mean()
#         surrogate_loss_network_input = torch.cat([filled_modality_reconstruction_losses, existing_modality_mask], dim=1)
#         surrogate_loss = self.surrogate_loss_network(surrogate_loss_network_input).mean()

#         return surrogate_loss

# class SurrogateLossNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()

#         in_features = 24 # 12 modalities, each with a reconstruction loss and a mask value
#         # in_features = 25 # 12 modalities, each with a reconstruction loss and a mask value
#         self.surrogate_loss_network = nn.Sequential(nn.Linear(in_features, 64),
#                                                     nn.GELU(),
#                                                     nn.Linear(64, 64),
#                                                     nn.GELU(),
#                                                     nn.Linear(64, 1),
#                                                     nn.Softplus()) # ensures the output is non-negative
#         # self.surrogate_loss_network = nn.Linear(in_features, 1, bias=False)

#         # with torch.no_grad():
#         #     self.surrogate_loss_network.weight.fill_(1.0 / in_features) # initializes all weights to 1/in_features

#     # def forward(self, modality_reconstruction_losses, initial_task_prediction):
#     def forward(self, modality_reconstruction_losses):
#         # if len(initial_task_prediction.shape) > 2:
#         #     initial_task_prediction = initial_task_prediction.mean(dim=(2,3)) # collapses the spatial dimensions for pixelwise tasks

#         stacked_modality_reconstruction_losses = torch.stack(list(modality_reconstruction_losses.values()), dim=1)
#         existing_modality_mask = (~stacked_modality_reconstruction_losses.isnan()).float() # creates a mask for existing modalities (not NaN)
#         filled_modality_reconstruction_losses = torch.nan_to_num(stacked_modality_reconstruction_losses, nan=0.0)
#         # surrogate_loss_network_input = torch.cat([filled_modality_reconstruction_losses, existing_modality_mask, initial_task_prediction], dim=1)
#         surrogate_loss_network_input = torch.cat([filled_modality_reconstruction_losses, existing_modality_mask], dim=1)
#         surrogate_loss = self.surrogate_loss_network(surrogate_loss_network_input).mean()

#         return surrogate_loss

# class SurrogateLossNetwork(nn.Module):
#     def __init__(self, in_features):
#         super().__init__()

#         hidden_dim = 128
#         self.surrogate_loss_network = nn.Sequential(nn.LayerNorm(in_features),
#                                                     nn.Linear(in_features, hidden_dim),
#                                                     nn.ReLU(),
#                                                     nn.Linear(hidden_dim, hidden_dim // 2),
#                                                     nn.ReLU(),
#                                                     nn.Linear(hidden_dim // 2, 1),
#                                                     nn.Softplus()) # ensures the output is non-negative

#     # def forward(self, embeddings, initial_task_prediction):
#     def forward(self, embeddings, modality_reconstruction_losses):
#         pooled_embeddings = embeddings.mean(dim=(2,3)) # global average pooling over the spatial dimensions
#         stacked_modality_reconstruction_losses = torch.stack(list(modality_reconstruction_losses.values()), dim=1)
#         existing_modality_mask = (~stacked_modality_reconstruction_losses.isnan()).float() # creates a mask for existing modalities (not NaN)
#         filled_modality_reconstruction_losses = torch.nan_to_num(stacked_modality_reconstruction_losses, nan=0.0)

#         # if len(initial_task_prediction.shape) > 2:
#         #     initial_task_prediction = initial_task_prediction.mean(dim=(2,3)) # collapses the spatial dimensions for pixelwise tasks

#         # surrogate_loss_network_input = torch.cat([pooled_embeddings, initial_task_prediction], dim=1)
#         # surrogate_loss = self.surrogate_loss_network(surrogate_loss_network_input).mean()
#         surrogate_loss_network_input = torch.cat([pooled_embeddings, filled_modality_reconstruction_losses, existing_modality_mask], dim=1)
#         surrogate_loss = self.surrogate_loss_network(surrogate_loss_network_input).mean()

#         return surrogate_loss

# class Hypernetwork(nn.Module):
#     def __init__(self, embedding_dim):
#         super().__init__()

#         in_features = 25 # 12 modalities, each with a reconstruction loss and a mask value
#         hidden_dim = 64
#         num_params_to_predict = 2 * embedding_dim # gamma and beta for each embedding dimension

#         self.hypernetwork = nn.Sequential(nn.Linear(in_features, hidden_dim),
#                                           nn.ReLU(),
#                                           nn.Linear(hidden_dim, num_params_to_predict))

#     def forward(self, modality_reconstruction_losses, initial_task_prediction):
#         if len(initial_task_prediction.shape) > 2:
#             initial_task_prediction = initial_task_prediction.mean(dim=(2,3)) # collapses the spatial dimensions for pixelwise tasks

#         stacked_modality_reconstruction_losses = torch.stack(list(modality_reconstruction_losses.values()), dim=1)
#         existing_modality_mask = (~stacked_modality_reconstruction_losses.isnan()).float() # creates a mask for existing modalities (not NaN)
#         filled_modality_reconstruction_losses = torch.nan_to_num(stacked_modality_reconstruction_losses, nan=0.0)
#         hypernetwork_input = torch.cat([filled_modality_reconstruction_losses, existing_modality_mask, initial_task_prediction], dim=1)
#         feature_adapter_parameters = self.hypernetwork(hypernetwork_input)

#         return feature_adapter_parameters

# class Hypernetwork(nn.Module):
#     def __init__(self, embedding_dim, in_features):
#         super().__init__()

#         hidden_dim = 64
#         num_params_to_predict = 2 * embedding_dim # gamma and beta for each embedding dimension

#         self.hypernetwork = nn.Sequential(nn.LayerNorm(in_features),
#                                           nn.Linear(in_features, hidden_dim),
#                                           nn.ReLU(),
#                                           nn.Linear(hidden_dim, num_params_to_predict))

#     def forward(self, embeddings, initial_task_prediction):
#         pooled_embeddings = embeddings.mean(dim=(2,3)) # global average pooling over the spatial dimensions

#         if len(initial_task_prediction.shape) > 2:
#             initial_task_prediction = initial_task_prediction.mean(dim=(2,3)) # collapses the spatial dimensions for pixelwise tasks

#         hypernetwork_input = torch.cat([pooled_embeddings, initial_task_prediction], dim=1)
#         feature_adapter_parameters = self.hypernetwork(hypernetwork_input)

#         return feature_adapter_parameters

# class FeatureAdapter(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, initial_input_embeddings, feature_adapter_parameters):
#         gamma, beta = feature_adapter_parameters.unsqueeze(-1).unsqueeze(-1).split(feature_adapter_parameters.size(1) // 2, dim=1)

#         return gamma * initial_input_embeddings + beta

class Adapter(nn.Module):
    """
    Unified residual adapter for [B,C,H,W] features.
    Blocks:
      - FiLM: per-channel scale/shift
      - Depthwise: 3x3 depthwise conv (spatially local)
      - LowRank: 1x1 -> r -> 1x1 bottleneck (channel mixing)

    y = z + alpha * ( w_film * FiLM(z) + w_dw * DW(z) + w_lr * LR(z) )

    All paths are identity-initialized (zero) so the adapter starts as no-op.
    """
    def __init__(self, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.alpha = nn.Parameter(torch.tensor(1e-1)) # global strength (learnable, starts small)
        self.path_logits = nn.Parameter(torch.zeros(3)) # per-path mixture weights (learnable, start equal)

        # --- FiLM path ---
        self.gamma = nn.Parameter(torch.zeros(embedding_dim)) # per-channel scale
        self.beta  = nn.Parameter(torch.zeros(embedding_dim)) # per-channel shift

        # --- Depthwise path ---
        self.depthwise_convolution = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding=1, groups=embedding_dim, bias=True)
        nn.init.zeros_(self.depthwise_convolution.weight) # zero init = identity
        nn.init.zeros_(self.depthwise_convolution.bias) # zero init = identity

        # --- Low-rank path (1x1 bottleneck) ---
        r = max(1, min(8, embedding_dim))
        self.pw1 = nn.Conv2d(embedding_dim, r, kernel_size=1, bias=False) # compresses the embedding dimension
        self.pw2 = nn.Conv2d(r, embedding_dim, kernel_size=1, bias=False) # expands back to the embedding dimension
        nn.init.zeros_(self.pw1.weight) # zero init = identity
        nn.init.zeros_(self.pw2.weight) # zero init = identity

        self.layer_norm = nn.GroupNorm(num_groups=1, num_channels=embedding_dim, affine=False) # LayerNorm2D surrogate

    def forward(self, initial_input_embeddings): # initial_input_embeddings: [B,C,H,W]
        paths = []
        g = self.gamma.view(1, -1, 1, 1)
        b = self.beta.view(1, -1, 1, 1)
        paths.append(g * initial_input_embeddings + b) # FiLM residual
        paths.append(self.depthwise_convolution(initial_input_embeddings)) # depthwise residual
        paths.append(self.pw2(self.pw1(initial_input_embeddings))) # low-rank residual
        w = torch.softmax(self.path_logits, dim=0) # softmax mixture over enabled paths
        residual = sum(w[i] * paths[i] for i in range(len(paths))) # sum weighted residuals (match len(paths))
        residual = self.layer_norm(residual) # light normalization of residual before adding

        return initial_input_embeddings + self.alpha * residual # global, learnable small step

class EncoderDecoder(nn.Module):
    def __init__(self, task, architecture, adaptation_mode, pixelwise, num_classes, pretrained, lr):
        super().__init__()

        self.architecture = architecture
        self.adaptation_mode = adaptation_mode
        self.pixelwise = pixelwise
        self.encoder = globals()[f'{architecture}Encoder'](pretrained, adaptation_mode)
        self.lr = lr
        self.embedding_dim = architecture_embedding_dims[architecture]

        # if adaptation_mode in ['multimodal_joint_training', 'ttt-mjt', 'multimodal_MT3', 'multimodal_sln']:
        #     self.embedding_dim += 320 # 320 is the embedding dimension of the task modality encoder

        # if 'TTT' in adaptation_mode or '-10' in adaptation_mode or '-20' in adaptation_mode or adaptation_mode in ['FT', 'LP', 'multimodal', 'JT', 'UDA-SS', 'multimodal_joint_training', 'MT3', 'multimodal_MT3', 'sln', 'multimodal_sln', 'joint_probing', 'sln_input_embeddings', 'sln_encode', 'MT3_metabatch', 'MT3_frozen', 'rna', 'rna_input_embeddings']:
        if 'TTT' in adaptation_mode or adaptation_mode in ['FT', 'JT', 'MT3_metabatch']:
            self.task_decoder = TaskDecoder(architecture, adaptation_mode, pixelwise, num_classes)

        # if adaptation_mode in ['multimodal', 'multimodal_joint_training', 'ttt-mjt', 'multimodal_MT3', 'multimodal_sln', 'sln_encode']:
        #     self.task_modality_encoder = TaskModalityEncoder()

        # if 'TTT' in adaptation_mode or adaptation_mode in ['JT', 'TMD', 'UDA-SS', 'TTT-UDA-SS', 'multimodal_joint_training', 'ttt-jt', 'uda-ttt', 'ttt-jt-10', 'ttt-jt-20', 'ttt-mjt', 'MT3', 'MT3-10', 'MT3-20', 'multimodal_MT3', 'sln', 'sln-10', 'sln-20', 'multimodal_sln', 'joint_probing', 'ttt-adapter', 'ttt-jp', 'MT3_metabatch', 'MT3_frozen', 'rna']:
        if 'TTT' in adaptation_mode or adaptation_mode in ['JT', 'TMD', 'MT3_metabatch']:
            self.task_modality_decoder = TaskModalityDecoder(self.embedding_dim)

        # if 'TTT' in adaptation_mode or adaptation_mode in ['JT', 'TTT', 'UDA-SS', 'TTT-UDA-SS', 'multimodal_joint_training', 'ttt-jt', 'uda-ttt', 'ttt-jt-10', 'ttt-jt-20', 'ttt-mjt', 'MT3', 'MT3-10', 'MT3-20', 'multimodal_MT3', 'sln', 'multimodal_sln', 'joint_probing',  'ttt-adapter', 'ttt-jp', 'MT3_metabatch', 'MT3_frozen']:
        if 'TTT' in adaptation_mode or adaptation_mode in ['JT', 'MT3_metabatch']:
            # self.task_modality_decoder_loss = TaskModalityDecoderLoss()
            self.modality_reconstruction_loss_calculator = ModalityReconstructionLossCalculator()

            # self.task_modality_decoder_loss = TaskModalityDecoderWeightedLoss()

        # if adaptation_mode in ['sln', 'sln-10', 'sln-20', 'multimodal_sln', 'rna']:
        #     self.modality_reconstruction_loss_calculator = ModalityReconstructionLossCalculator()

        if '-adapter' in adaptation_mode:
            self.adapter = Adapter(embedding_dim=self.embedding_dim)
            self.adapter.requires_grad_(False) # freezes the adapter

        # elif adaptation_mode in ['sln', 'sln-10', 'sln-20']:
        #     self.surrogate_loss_network = SurrogateLossNetwork()
        # elif adaptation_mode in ['multimodal_sln']:
        #     self.surrogate_loss_network = SurrogateLossNetwork(in_features=self.embedding_dim+24) # 320 is the embedding dimension of the task modality encoder
        # elif adaptation_mode in ['sln_input_embeddings']:
        #     self.surrogate_loss_network = SurrogateLossNetwork(in_features=self.embedding_dim+1)
        # elif adaptation_mode == 'sln_encode':
        #     self.surrogate_loss_network = SurrogateLossNetwork(in_features=self.embedding_dim+2048+1)
        # elif adaptation_mode == 'rna':
        #     self.hypernetwork = Hypernetwork(embedding_dim=self.embedding_dim)
        #     self.feature_adapter = FeatureAdapter()
        # elif adaptation_mode == 'rna_input_embeddings':
        #     self.hypernetwork = Hypernetwork(embedding_dim=self.embedding_dim, in_features=self.embedding_dim+1)
        #     self.feature_adapter = FeatureAdapter()

        # load model from checkpoint
        # if 'TTT' in adaptation_mode or '-10' in adaptation_mode or '-20' in adaptation_mode or adaptation_mode in ['TMD', 'sln_input_embeddings', 'sln_encode', 'MT3_frozen', 'rna', 'rna_input_embeddings']:
        if 'TTT' in adaptation_mode or adaptation_mode == 'TMD':
            if adaptation_mode == 'TMD':
                state_dict = get_state_dict(task, architecture, 'FT')
            # elif 'TTT' in adaptation_mode or adaptation_mode in ['ttt-jt', 'ttt-jt-10', 'ttt-jt-20', 'ttt-adapter', 'MT3_frozen', 'rna']:
            elif 'FT-TTT' in adaptation_mode:
                state_dict = get_state_dict(task, architecture, 'FT')
            elif 'JT-TTT' in adaptation_mode:
                state_dict = get_state_dict(task, architecture, 'JT')
            elif 'MT3-TTT' in adaptation_mode:
                state_dict = get_state_dict(task, architecture, 'MT3_metabatch')
            # elif 'TTT' in adaptation_mode:
            #     # state_dict = get_state_dict(task, architecture, 'JT')
            #     # state_dict = get_state_dict(task, architecture, 'TMD')
            #     state_dict = get_state_dict(task, architecture, 'FT')
            # elif adaptation_mode == 'TTT-UDA-SS':
            #     state_dict = get_state_dict(task, architecture, 'UDA-SS')
            # elif adaptation_mode in ['sln-10', 'sln-20']:
            #     state_dict = get_state_dict(task, architecture, 'sln')
            # elif adaptation_mode == 'ttt-jp':
            #     state_dict = get_state_dict(task, architecture, 'joint_probing')
            # elif adaptation_mode == 'ttt-mjt':
            #     state_dict = get_state_dict(task, architecture, 'multimodal_joint_training')
            # elif adaptation_mode in ['uda-ttt', 'sln_input_embeddings', 'rna_input_embeddings']:
            #     # state_dict = get_state_dict(task, architecture, 'FT')
            #     state_dict = get_state_dict(task, architecture, 'standard')
            # elif adaptation_mode == 'sln_encode':
            #     state_dict = get_state_dict(task, architecture, 'multimodal')

            # load encoder weights
            encoder_state_dict = {key.removeprefix('model.encoder.'): value for key, value in state_dict.items() if key.startswith('model.encoder')} # filters the state_dict to only include the encoder parameters
            self.encoder.load_state_dict(encoder_state_dict)

        # freeze encoder
        # if 'TTT' in adaptation_mode or '-10' in adaptation_mode or '-20' in adaptation_mode or adaptation_mode in ['TMD', 'LP', 'sln_input_embeddings', 'sln_encode', 'MT3_frozen', 'rna', 'rna_input_embeddings', 'joint_probing']:
        if 'TTT' in adaptation_mode or adaptation_mode == 'TMD':
            self.encoder.requires_grad_(False) # freezes the encoder

        # freeze task decoder
        # if 'TTT' in adaptation_mode or '-10' in adaptation_mode or '-20' in adaptation_mode or adaptation_mode in ['sln_input_embeddings', 'sln_encode', 'MT3_frozen', 'rna', 'rna_input_embeddings']:
        if 'TTT' in adaptation_mode:
            # load task decoder weights
            # ft_state_dict = get_state_dict(task, architecture, 'FT')
            # task_decoder_state_dict = {key.removeprefix('model.task_decoder.'): value for key, value in ft_state_dict.items() if key.startswith('model.task_decoder')} # filters the state_dict to only include the decoder parameters
            task_decoder_state_dict = {key.removeprefix('model.task_decoder.'): value for key, value in state_dict.items() if key.startswith('model.task_decoder')} # filters the state_dict to only include the decoder parameters
            self.task_decoder.load_state_dict(task_decoder_state_dict)
            self.task_decoder.requires_grad_(False) # freezes the task decoder

            # load task modality decoder weights
            if 'FT-TTT' in adaptation_mode:
                tmd_state_dict = get_state_dict(task, architecture, 'TMD')
                task_modality_decoder_state_dict = {key.removeprefix('model.task_modality_decoder.'): value for key, value in tmd_state_dict.items() if key.startswith('model.task_modality_decoder')}
            else:
                task_modality_decoder_state_dict = {key.removeprefix('model.task_modality_decoder.'): value for key, value in state_dict.items() if key.startswith('model.task_modality_decoder')}

            self.task_modality_decoder.load_state_dict(task_modality_decoder_state_dict)

            # if adaptation_mode not in ['uda-ttt', 'sln_input_embeddings', 'sln_encode', 'rna_input_embeddings']:
            #     # load task modality decoder weights
            #     # tmd_state_dict = get_state_dict(task, architecture, 'TMD')
            #     task_modality_decoder_state_dict = {key.removeprefix('model.task_modality_decoder.'): value for key, value in state_dict.items() if key.startswith('model.task_modality_decoder')}
            #     # task_modality_decoder_state_dict = {key.removeprefix('model.task_modality_decoder.'): value for key, value in tmd_state_dict.items() if key.startswith('model.task_modality_decoder')}
            #     self.task_modality_decoder.load_state_dict(task_modality_decoder_state_dict)

        # freeze task modality decoder
        # if 'TTT' in adaptation_mode or 'ttt' in adaptation_mode or '-10' in adaptation_mode or '-20' in adaptation_mode or adaptation_mode in ['rna']:
        if 'TTT' in adaptation_mode:
            self.task_modality_decoder.requires_grad_(False) # freezes the task modality decoder
            self.mode = 'val'
            # self.val_best_num_iterations = 50
            self.val_best_num_iterations = 5

            # self.iteration_performances = []

        # # freeze task modality encoder
        # if adaptation_mode in ['ttt-mjt', 'sln_encode']:
        #     # load task modality encoder weights
        #     task_modality_encoder_state_dict = {key.removeprefix('model.task_modality_encoder.'): value for key, value in state_dict.items() if key.startswith('model.task_modality_encoder')} # filters the state_dict to only include the task modality encoder parameters
        #     self.task_modality_encoder.load_state_dict(task_modality_encoder_state_dict)
        #     self.task_modality_encoder.requires_grad_(False) # freezes the task modality encoder

        # if adaptation_mode in ['sln-10', 'sln-20']:
        #     # load surrogate loss network weights
        #     surrogate_loss_network_state_dict = {key.removeprefix('model.surrogate_loss_network.'): value for key, value in state_dict.items() if key.startswith('model.surrogate_loss_network')} # filters the state_dict to only include the surrogate loss network parameters
        #     self.surrogate_loss_network.load_state_dict(surrogate_loss_network_state_dict)
        #     self.surrogate_loss_network.requires_grad_(False) # freezes the surrogate loss network

    def upsample_modality_embeddings(self, modality_embeddings, input_embeddings):
        if modality_embeddings.shape[2] != input_embeddings.shape[2]:
            modality_embeddings = nn.Upsample(size=input_embeddings.shape[2], mode='bilinear')(modality_embeddings) # upsamples bilinearly to match the spatial dimensions of the input embeddings

        return modality_embeddings

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
                del modality_reconstructions
                gradients_per_modality = []

                for modality_loss in modality_reconstruction_losses.values():
                    if not modality_loss.isnan().all():
                        gradients_per_modality.append(torch.autograd.grad(modality_loss.nanmean(), encoder_parameters.values(), retain_graph=True))

                del modality_reconstruction_losses
                normalized_gradients_per_modality = []

                for modality_grads in gradients_per_modality:
                    grad_norm = torch.linalg.vector_norm(torch.cat([g.flatten() for g in modality_grads]))
                    normalized_gradients_per_modality.append(tuple(g / (grad_norm + 1e-6) for g in modality_grads))

                del gradients_per_modality
                del grad_norm
                averaged_grads = []

                for param_idx in range(len(normalized_gradients_per_modality[0])):
                    # Stack and average gradients for this parameter
                    param_grads_across_modalities = [normalized_gradients_per_modality[m][param_idx] for m in range(len(normalized_gradients_per_modality))]
                    averaged_grad = torch.stack(param_grads_across_modalities).mean(dim=0) # average element-wise
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

    # def mt3(self, input_data, task_modality_data, num_iterations):
    #     self.encoder.eval()
    #     self.task_decoder.eval()
    #     self.task_modality_decoder.eval()

    #     with sdpa_kernel(SDPBackend.MATH):
    #         iteration_predictions = []

    #         with torch.enable_grad(): # need to be able to compute gradients even during validation and testing
    #             encoder_parameters = {name: parameter.detach().clone().requires_grad_() for name, parameter in self.encoder.named_parameters()}

    #             for i in range(num_iterations+1): # iterations
    #                 input_embeddings = functional_call(self.encoder, encoder_parameters, (input_data,))
    #                 task_prediction = self.task_decoder(input_embeddings)
    #                 iteration_predictions.append(task_prediction)

    #                 if i == num_iterations:
    #                     break

    #                 task_modality_reconstruction_loss = self.task_modality_decoder_loss(modality_reconstructions=self.task_modality_decoder(input_embeddings), task_modality_data=task_modality_data)
    #                 task_modality_reconstruction_loss_grads = torch.autograd.grad(task_modality_reconstruction_loss, encoder_parameters.values(), create_graph=True) # computes the gradients of the task modality reconstruction loss with respect to the encoder parameters

    #                 with torch.no_grad():
    #                     encoder_parameters = {name: (parameter - self.lr * grad).detach().requires_grad_() for (name, parameter), grad in zip(encoder_parameters.items(), task_modality_reconstruction_loss_grads)}

    #         iteration_predictions = torch.stack(iteration_predictions) # (num_iterations+1, batch_size, num_classes, optional height, optional width)

    #         return iteration_predictions

    def mt3(self, input_data, task_modality_data, num_iterations, return_all_iterations):
        self.encoder.eval()
        self.task_decoder.eval()
        self.task_modality_decoder.eval()

        with sdpa_kernel(SDPBackend.MATH):
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

                    task_modality_reconstruction_loss = self.task_modality_decoder_loss(modality_reconstructions=self.task_modality_decoder(input_embeddings), task_modality_data=task_modality_data)
                    del input_embeddings
                    task_modality_reconstruction_loss_grads = torch.autograd.grad(task_modality_reconstruction_loss, encoder_parameters.values(), create_graph=True) # computes the gradients of the task modality reconstruction loss with respect to the encoder parameters
                    del task_modality_reconstruction_loss
                    # grad_norm = torch.linalg.vector_norm(torch.stack([g.detach().float().norm() for g in task_modality_reconstruction_loss_grads]))
                    # max_norm = 1
                    # scale = (max_norm / (grad_norm + 1e-6)).clamp(max=1.0)

                    with torch.no_grad():
                        # encoder_parameters = {name: (parameter - self.lr * scale * grad).detach().requires_grad_() for (name, parameter), grad in zip(encoder_parameters.items(), task_modality_reconstruction_loss_grads)}
                        encoder_parameters = {name: (parameter - self.lr * grad).detach().requires_grad_() for (name, parameter), grad in zip(encoder_parameters.items(), task_modality_reconstruction_loss_grads)}

                    # del task_modality_reconstruction_loss_grads, grad_norm, scale
                    del task_modality_reconstruction_loss_grads

    # def mt3_metabatch(self, input_data, task_modality_data, num_iterations):
    #     self.encoder.eval()
    #     self.task_decoder.eval()
    #     self.task_modality_decoder.eval()

    #     with sdpa_kernel(SDPBackend.MATH):
    #         iteration_predictions = []

    #         with torch.enable_grad(): # need to be able to compute gradients even during validation and testing
    #             encoder_parameters = {name: parameter.detach().clone().requires_grad_() for name, parameter in self.encoder.named_parameters()}

    #             for i in range(num_iterations+1): # iterations
    #                 mini_batch_iteration_predictions = []

    #                 for i in range(len(list(input_data.values())[0])): # for each mini batch
    #                     mini_batch_input_data = {key: value[i].unsqueeze(0) for key, value in input_data.items()}
    #                     mini_batch_task_modality_data = {key: {k: v[i].unsqueeze(0) for k, v in data.items()} for key, data in task_modality_data.items()}
    #                     input_embeddings = functional_call(self.encoder, encoder_parameters, (mini_batch_input_data,))
    #                     task_prediction = self.task_decoder(input_embeddings)
    #                     mini_batch_iteration_predictions.append(task_prediction)

    #                     if i == num_iterations:
    #                         break

    #                     task_modality_reconstruction_loss = self.task_modality_decoder_loss(modality_reconstructions=self.task_modality_decoder(input_embeddings), task_modality_data=mini_batch_task_modality_data)
    #                     task_modality_reconstruction_loss_grads = torch.autograd.grad(task_modality_reconstruction_loss, encoder_parameters.values(), create_graph=True) # computes the gradients of the task modality reconstruction loss with respect to the encoder parameters

    #                     with torch.no_grad():
    #                         encoder_parameters = {name: (parameter - self.lr * grad).detach().requires_grad_() for (name, parameter), grad in zip(encoder_parameters.items(), task_modality_reconstruction_loss_grads)}

    #             iteration_predictions.append(torch.stack(mini_batch_iteration_predictions))

    #         iteration_predictions = torch.stack(iteration_predictions)

    #         return iteration_predictions

    def sln(self, input_data, task_modality_data, num_iterations):
        self.encoder.eval()
        self.task_decoder.eval()
        self.task_modality_decoder.eval()

        with sdpa_kernel(SDPBackend.MATH):
            iteration_predictions = []

            with torch.enable_grad(): # need to be able to compute gradients even during validation and testing
                encoder_parameters = {name: parameter.detach().clone().requires_grad_() for name, parameter in self.encoder.named_parameters()}

                for i in range(num_iterations+1): # iterations
                    input_embeddings = functional_call(self.encoder, encoder_parameters, (input_data,))
                    task_prediction = self.task_decoder(input_embeddings)
                    iteration_predictions.append(task_prediction)

                    if i == num_iterations:
                        break

                    surrogate_loss = self.surrogate_loss_network(self.modality_reconstruction_loss_calculator(modality_reconstructions=self.task_modality_decoder(input_embeddings), task_modality_data=task_modality_data))
                    surrogate_loss_grads = torch.autograd.grad(surrogate_loss, encoder_parameters.values(), create_graph=True) # computes the gradients of the surrogate loss with respect to the encoder parameters

                    with torch.no_grad():
                        encoder_parameters = {name: (parameter - self.lr * grad).detach().requires_grad_() for (name, parameter), grad in zip(encoder_parameters.items(), surrogate_loss_grads)}

            iteration_predictions = torch.stack(iteration_predictions) # (num_iterations+1, batch_size, num_classes, optional height, optional width)

            return iteration_predictions

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
        # elif self.adaptation_mode in ['TTT', 'TTT-Geo', 'ttt-jt', 'TTT-UDA-SS', 'uda-ttt', 'ttt-jp']:
        # elif self.adaptation_mode in ['TTT', 'TTT-Geo']:
        # elif any(string in self.adaptation_mode for string in ['TTT', 'TTT-Geo']):
        elif 'MT3-TTT' in self.adaptation_mode:
            # num_iterations = int(self.adaptation_mode.split('-')[-1])
            print(f'Running MT3-TTT for {self.val_best_num_iterations} iteration(s)')
            task_prediction = self.mt3(input_data, task_modality_data, num_iterations=self.val_best_num_iterations, return_all_iterations=self.mode=='val')

            return task_prediction
        elif '-adapter' in self.adaptation_mode:
            print(f'Running TTT adapter for {self.val_best_num_iterations} iteration(s)')
            task_prediction = self.ttt_adapter(input_data, task_modality_data, num_iterations=self.val_best_num_iterations, return_all_iterations=self.mode=='val')

            return task_prediction
        elif 'TTT' in self.adaptation_mode:
            print(f'Running TTT for {self.val_best_num_iterations} iteration(s)')
            task_prediction = self.ttt(input_data, task_modality_data, num_iterations=self.val_best_num_iterations, return_all_iterations=self.mode=='val')

            return task_prediction

            # iteration_predictions = self.ttt(input_data, task_modality_data, num_iterations=1)

            # return iteration_predictions
        # elif self.adaptation_mode in ['TTT-10', 'TTT-Geo-10', 'ttt-jt-10', 'uda-ttt-10']:
        #     iteration_predictions = self.ttt(input_data, task_modality_data, num_iterations=10)

        #     return iteration_predictions
        # elif self.adaptation_mode in ['TTT-20', 'ttt-jt-20']:
        #     iteration_predictions = self.ttt(input_data, task_modality_data, num_iterations=20)

        #     return iteration_predictions
        # elif self.adaptation_mode in ['TTT-30']:
        #     iteration_predictions = self.ttt(input_data, task_modality_data, num_iterations=30)

        #     return iteration_predictions
        # elif self.adaptation_mode in ['TTT-40']:
        #     iteration_predictions = self.ttt(input_data, task_modality_data, num_iterations=40)

        #     return iteration_predictions
        # elif self.adaptation_mode in ['TTT-50', 'TTT-Geo-50']:
        #     iteration_predictions = self.ttt(input_data, task_modality_data, num_iterations=50)

        #     return iteration_predictions
        elif self.adaptation_mode == 'multimodal':
            input_embeddings = self.encoder(input_data)
            modality_embeddings = self.task_modality_encoder(task_modality_data)
            modality_embeddings = self.upsample_modality_embeddings(modality_embeddings, input_embeddings)
            concatenated_embeddings = torch.cat([input_embeddings, modality_embeddings], dim=1)
            task_prediction = self.task_decoder(concatenated_embeddings)

            return task_prediction
        elif self.adaptation_mode == 'UDA-SS':
            input_embeddings = self.encoder(input_data)
            task_prediction = self.task_decoder(input_embeddings)
            modality_reconstructions = self.task_modality_decoder(input_embeddings)
            task_modality_reconstruction_loss = self.task_modality_decoder_loss(modality_reconstructions, task_modality_data)

            return input_embeddings, task_prediction, modality_reconstructions, task_modality_reconstruction_loss
        elif self.adaptation_mode == 'multimodal_joint_training':
            input_embeddings = self.encoder(input_data)
            modality_embeddings = self.task_modality_encoder(task_modality_data)
            modality_embeddings = self.upsample_modality_embeddings(modality_embeddings, input_embeddings)
            concatenated_embeddings = torch.cat([input_embeddings, modality_embeddings], dim=1)
            task_prediction = self.task_decoder(concatenated_embeddings) if not self.pixelwise else self.task_decoder(concatenated_embeddings, input_data)
            modality_reconstructions = self.task_modality_decoder(concatenated_embeddings, input_data)
            task_modality_reconstruction_loss = self.task_modality_decoder_loss(modality_reconstructions, task_modality_data)

            return task_prediction, modality_reconstructions, task_modality_reconstruction_loss
        elif self.adaptation_mode == 'joint_probing':
            self.encoder.eval()

            input_embeddings = self.encoder(input_data)
            task_prediction = self.task_decoder(input_embeddings) if not self.pixelwise else self.task_decoder(input_embeddings, input_data)
            modality_reconstructions = self.task_modality_decoder(input_embeddings, input_data)
            task_modality_reconstruction_loss = self.task_modality_decoder_loss(modality_reconstructions, task_modality_data)

            return task_prediction, modality_reconstructions, task_modality_reconstruction_loss
        elif self.adaptation_mode in ['ttt-mjt']:
            self.encoder.eval()
            self.task_decoder.eval()
            self.task_modality_encoder.eval()
            self.task_modality_decoder.eval()

            iteration_predictions = []

            with torch.enable_grad(): # need to be able to compute gradients even during validation and testing
                encoder_parameters = {name: parameter.detach().clone().requires_grad_() for name, parameter in self.encoder.named_parameters()}
                task_modality_encoder_parameters = {name: parameter.detach().clone().requires_grad_() for name, parameter in self.task_modality_encoder.named_parameters()}

                for _ in range(2): # iterations
                    input_embeddings = functional_call(self.encoder, encoder_parameters, (input_data,))
                    modality_embeddings = functional_call(self.task_modality_encoder, task_modality_encoder_parameters, (task_modality_data,))

                    if modality_embeddings.shape[2] != input_embeddings.shape[2]:
                        modality_embeddings = nn.Upsample(size=input_embeddings.shape[2], mode='bilinear')(modality_embeddings)

                    concatenated_embeddings = torch.cat([input_embeddings, modality_embeddings], dim=1)
                    task_prediction = self.task_decoder(concatenated_embeddings) if not self.pixelwise else self.task_decoder(concatenated_embeddings, input_data)
                    iteration_predictions.append(task_prediction)
                    task_modality_reconstruction_loss = self.task_modality_decoder_loss(modality_reconstructions=self.task_modality_decoder(concatenated_embeddings, input_data), task_modality_data=task_modality_data)
                    grads = torch.autograd.grad(task_modality_reconstruction_loss, list(encoder_parameters.values()) + list(task_modality_encoder_parameters.values())) # computes the gradients of the task modality reconstruction loss with respect to the encoder and task modality encoder parameters
                    # task_modality_reconstruction_loss_encoder_grads = torch.autograd.grad(task_modality_reconstruction_loss, encoder_parameters.values(), retain_graph=True) # computes the gradients of the task modality reconstruction loss with respect to the encoder parameters
                    # task_modality_reconstruction_loss_task_modality_encoder_grads = torch.autograd.grad(task_modality_reconstruction_loss, task_modality_encoder_parameters.values()) # computes the gradients of the task modality reconstruction loss with respect to the encoder parameters
                    task_modality_reconstruction_loss_encoder_grads = grads[:len(encoder_parameters)]
                    task_modality_reconstruction_loss_task_modality_encoder_grads = grads[len(encoder_parameters):]

                    with torch.no_grad():
                        encoder_parameters = {name: param - self.lr * grad for (name, param), grad in zip(encoder_parameters.items(), task_modality_reconstruction_loss_encoder_grads)}
                        task_modality_encoder_parameters = {name: param - self.lr * grad for (name, param), grad in zip(task_modality_encoder_parameters.items(), task_modality_reconstruction_loss_task_modality_encoder_grads)}

                    encoder_parameters = {name: parameter.detach().requires_grad_() for name, parameter in encoder_parameters.items()} # re-attaches the computation graph for the next iteration
                    task_modality_encoder_parameters = {name: parameter.detach().requires_grad_() for name, parameter in task_modality_encoder_parameters.items()} # re-attaches the computation graph for the next iteration

            iteration_predictions = torch.stack(iteration_predictions) # (num_iterations, batch_size, num_classes, optional height, optional width)

            return iteration_predictions
        # elif self.adaptation_mode == 'ttt-adapter':
        #     self.encoder.eval()
        #     self.task_decoder.eval()
        #     self.task_modality_decoder.eval()

        #     iteration_predictions = []

        #     with torch.enable_grad(): # need to be able to compute gradients even during validation and testing
        #         adapter_parameters = {name: parameter.detach().clone().requires_grad_() for name, parameter in self.adapter.named_parameters()}
        #         task_modality_decoder_parameters = {name: parameter.detach().clone().requires_grad_() for name, parameter in self.task_modality_decoder.named_parameters()}
        #         input_embeddings = self.encoder(input_data)

        #         for _ in range(11): # iterations
        #             adapted_embeddings = functional_call(self.adapter, adapter_parameters, (input_embeddings,))
        #             task_prediction = self.task_decoder(adapted_embeddings) if not self.pixelwise else self.task_decoder(adapted_embeddings, input_data)
        #             iteration_predictions.append(task_prediction)
        #             modality_reconstructions = functional_call(self.task_modality_decoder, task_modality_decoder_parameters, (adapted_embeddings, input_data))
        #             task_modality_reconstruction_loss = self.task_modality_decoder_loss(modality_reconstructions, task_modality_data)
        #             task_modality_reconstruction_loss_grads = torch.autograd.grad(task_modality_reconstruction_loss, adapter_parameters.values(), retain_graph=True) # computes the gradients of the task modality reconstruction loss with respect to the adapter parameters
        #             task_modality_reconstruction_loss_task_modality_decoder_grads = torch.autograd.grad(task_modality_reconstruction_loss, task_modality_decoder_parameters.values(), retain_graph=True) # computes the gradients of the task modality reconstruction loss with respect to the encoder parameters

        #             with torch.no_grad():
        #                 adapter_parameters = {name: param - self.lr * grad for (name, param), grad in zip(adapter_parameters.items(), task_modality_reconstruction_loss_grads)}
        #                 task_modality_decoder_parameters = {name: param - self.lr * grad for (name, param), grad in zip(task_modality_decoder_parameters.items(), task_modality_reconstruction_loss_task_modality_decoder_grads)}

        #             adapter_parameters = {name: parameter.detach().requires_grad_() for name, parameter in adapter_parameters.items()} # re-attaches the computation graph for the next iteration
        #             task_modality_decoder_parameters = {name: parameter.detach().requires_grad_() for name, parameter in task_modality_decoder_parameters.items()} # re-attaches the computation graph for the next iteration

        #     iteration_predictions = torch.stack([initial_task_prediction, adapted_task_prediction]) # (num_iterations, batch_size, num_classes, optional height, optional width)

        #     return iteration_predictions
        elif self.adaptation_mode == 'MT3':
            with sdpa_kernel(SDPBackend.MATH):
                with torch.enable_grad(): # need to be able to compute gradients even during validation and testing
                    encoder_parameters = {name: parameter for name, parameter in self.encoder.named_parameters()}
                    initial_input_embeddings = functional_call(self.encoder, encoder_parameters, (input_data,))
                    initial_task_prediction = self.task_decoder(initial_input_embeddings)
                    task_modality_decoder_parameters = {name: parameter for name, parameter in self.task_modality_decoder.named_parameters()}
                    modality_reconstructions = functional_call(self.task_modality_decoder, task_modality_decoder_parameters, (initial_input_embeddings,))
                    task_modality_reconstruction_loss = self.task_modality_decoder_loss(modality_reconstructions, task_modality_data)
                    task_modality_reconstruction_loss_encoder_grads = torch.autograd.grad(task_modality_reconstruction_loss, encoder_parameters.values(), create_graph=True) # computes the gradients of the task modality reconstruction loss with respect to the encoder parameters
                    task_modality_reconstruction_loss_task_modality_decoder_grads = torch.autograd.grad(task_modality_reconstruction_loss, task_modality_decoder_parameters.values(), create_graph=True) # computes the gradients of the task modality reconstruction loss with respect to the encoder parameters
                    adapted_encoder_parameters = {name: param - self.lr * grad for (name, param), grad in zip(encoder_parameters.items(), task_modality_reconstruction_loss_encoder_grads)}
                    adapted_task_modality_decoder_parameters = {name: param - self.lr * grad for (name, param), grad in zip(task_modality_decoder_parameters.items(), task_modality_reconstruction_loss_task_modality_decoder_grads)}
                    adapted_input_embeddings = functional_call(self.encoder, adapted_encoder_parameters, (input_data,))
                    adapted_task_prediction = self.task_decoder(adapted_input_embeddings)
                    adapted_modality_reconstructions = functional_call(self.task_modality_decoder, adapted_task_modality_decoder_parameters, (adapted_input_embeddings,))
                    adapted_task_modality_reconstruction_loss = self.task_modality_decoder_loss(adapted_modality_reconstructions, task_modality_data)
                    iteration_predictions = torch.stack([initial_task_prediction, adapted_task_prediction]) # (num_iterations, batch_size, num_classes, optional height, optional width)

                    return iteration_predictions, adapted_modality_reconstructions, adapted_task_modality_reconstruction_loss
        elif self.adaptation_mode == 'MT3-10':
            iteration_predictions = self.mt3(input_data, task_modality_data, num_iterations=10)

            return iteration_predictions
        elif self.adaptation_mode == 'MT3-20':
            iteration_predictions = self.mt3(input_data, task_modality_data, num_iterations=20)

            return iteration_predictions
        elif self.adaptation_mode == 'multimodal_MT3':
            with sdpa_kernel(SDPBackend.MATH):
                with torch.enable_grad(): # need to be able to compute gradients even during validation and testing
                    encoder_parameters = {name: parameter for name, parameter in self.encoder.named_parameters()}
                    initial_input_embeddings = functional_call(self.encoder, encoder_parameters, (input_data,))
                    task_modality_encoder_parameters = {name: parameter for name, parameter in self.task_modality_encoder.named_parameters()}
                    initial_modality_embeddings = functional_call(self.task_modality_encoder, task_modality_encoder_parameters, (task_modality_data,))

                    if initial_modality_embeddings.shape[2] != initial_input_embeddings.shape[2]:
                        initial_modality_embeddings = nn.Upsample(size=initial_input_embeddings.shape[2], mode='bilinear')(initial_modality_embeddings) # upsamples bilinearly to match the spatial dimensions of the input embeddings

                    initial_concatenated_embeddings = torch.cat([initial_input_embeddings, initial_modality_embeddings], dim=1)
                    initial_task_prediction = self.task_decoder(initial_concatenated_embeddings) if not self.pixelwise else self.task_decoder(initial_concatenated_embeddings, input_data)
                    task_modality_decoder_parameters = {name: parameter for name, parameter in self.task_modality_decoder.named_parameters()}
                    modality_reconstructions = functional_call(self.task_modality_decoder, task_modality_decoder_parameters, (initial_concatenated_embeddings, input_data))
                    task_modality_reconstruction_loss = self.task_modality_decoder_loss(modality_reconstructions, task_modality_data)
                    task_modality_reconstruction_loss_encoder_grads = torch.autograd.grad(task_modality_reconstruction_loss, encoder_parameters.values(), create_graph=True) # computes the gradients of the task modality reconstruction loss with respect to the encoder parameters
                    task_modality_reconstruction_loss_task_modality_encoder_grads = torch.autograd.grad(task_modality_reconstruction_loss, task_modality_encoder_parameters.values(), create_graph=True) # computes the gradients of the task modality reconstruction loss with respect to the encoder parameters
                    task_modality_reconstruction_loss_task_modality_decoder_grads = torch.autograd.grad(task_modality_reconstruction_loss, task_modality_decoder_parameters.values(), create_graph=True) # computes the gradients of the task modality reconstruction loss with respect to the encoder parameters
                    adapted_encoder_parameters = {name: param - self.lr * grad for (name, param), grad in zip(encoder_parameters.items(), task_modality_reconstruction_loss_encoder_grads)}
                    adapted_task_modality_encoder_parameters = {name: param - self.lr * grad for (name, param), grad in zip(task_modality_encoder_parameters.items(), task_modality_reconstruction_loss_task_modality_encoder_grads)}
                    adapted_task_modality_decoder_parameters = {name: param - self.lr * grad for (name, param), grad in zip(task_modality_decoder_parameters.items(), task_modality_reconstruction_loss_task_modality_decoder_grads)}
                    adapted_input_embeddings = functional_call(self.encoder, adapted_encoder_parameters, (input_data,))
                    adapted_modality_embeddings = functional_call(self.task_modality_encoder, adapted_task_modality_encoder_parameters, (task_modality_data,))

                    if adapted_modality_embeddings.shape[2] != adapted_input_embeddings.shape[2]:
                        adapted_modality_embeddings = nn.Upsample(size=adapted_input_embeddings.shape[2], mode='bilinear')(adapted_modality_embeddings) # upsamples bilinearly to match the spatial dimensions of the input embeddings

                    adapted_concatenated_embeddings = torch.cat([adapted_input_embeddings, adapted_modality_embeddings], dim=1)
                    adapted_task_prediction = self.task_decoder(adapted_concatenated_embeddings) if not self.pixelwise else self.task_decoder(adapted_concatenated_embeddings, input_data)
                    adapted_modality_reconstructions = functional_call(self.task_modality_decoder, adapted_task_modality_decoder_parameters, (adapted_concatenated_embeddings, input_data))
                    adapted_task_modality_reconstruction_loss = self.task_modality_decoder_loss(adapted_modality_reconstructions, task_modality_data)
                    iteration_predictions = torch.stack([initial_task_prediction, adapted_task_prediction]) # (num_iterations, batch_size, num_classes, optional height, optional width)

                    return iteration_predictions, adapted_modality_reconstructions, adapted_task_modality_reconstruction_loss
        elif self.adaptation_mode == 'MT3_metabatch':
            with sdpa_kernel(SDPBackend.MATH):
                with torch.enable_grad(): # need to be able to compute gradients even during validation and testing
                    encoder_parameters = {name: parameter for name, parameter in self.encoder.named_parameters()}
                    task_modality_decoder_parameters = {name: parameter for name, parameter in self.task_modality_decoder.named_parameters()}

                    adapted_task_predictions = []
                    all_adapted_modality_reconstructions = []
                    adapted_task_modality_reconstruction_losses = []

                    for i in range(len(list(input_data.values())[0])): # for each mini batch
                        mini_batch_input_data = {key: value[i].unsqueeze(0) for key, value in input_data.items()}
                        mini_batch_task_modality_data = {key: {k: v[i].unsqueeze(0) for k, v in data.items()} for key, data in task_modality_data.items()}
                        initial_input_embeddings = functional_call(self.encoder, encoder_parameters, (mini_batch_input_data,))
                        modality_reconstructions = functional_call(self.task_modality_decoder, task_modality_decoder_parameters, (initial_input_embeddings,))
                        task_modality_reconstruction_loss = self.task_modality_decoder_loss(modality_reconstructions, mini_batch_task_modality_data)
                        task_modality_reconstruction_loss_encoder_grads = torch.autograd.grad(task_modality_reconstruction_loss, encoder_parameters.values(), create_graph=True) # computes the gradients of the task modality reconstruction loss with respect to the encoder parameters
                        task_modality_reconstruction_loss_task_modality_decoder_grads = torch.autograd.grad(task_modality_reconstruction_loss, task_modality_decoder_parameters.values(), create_graph=True) # computes the gradients of the task modality reconstruction loss with respect to the encoder parameters
                        adapted_encoder_parameters = {name: param - self.lr * grad for (name, param), grad in zip(encoder_parameters.items(), task_modality_reconstruction_loss_encoder_grads)}
                        adapted_task_modality_decoder_parameters = {name: param - self.lr * grad for (name, param), grad in zip(task_modality_decoder_parameters.items(), task_modality_reconstruction_loss_task_modality_decoder_grads)}
                        adapted_input_embeddings = functional_call(self.encoder, adapted_encoder_parameters, (mini_batch_input_data,))
                        adapted_task_prediction = self.task_decoder(adapted_input_embeddings)
                        adapted_modality_reconstructions = functional_call(self.task_modality_decoder, adapted_task_modality_decoder_parameters, (adapted_input_embeddings,))
                        adapted_task_modality_reconstruction_loss = self.task_modality_decoder_loss(adapted_modality_reconstructions, mini_batch_task_modality_data)

                        adapted_task_predictions.append(adapted_task_prediction)
                        all_adapted_modality_reconstructions.append(adapted_modality_reconstructions)
                        adapted_task_modality_reconstruction_losses.append(adapted_task_modality_reconstruction_loss)

                    adapted_task_predictions = torch.stack(adapted_task_predictions).squeeze(1)
                    all_adapted_modality_reconstructions = {modality: torch.stack([all_adapted_modality_reconstructions[i][modality].squeeze(0) for i in range(len(all_adapted_modality_reconstructions))]).squeeze(-1) for modality in all_adapted_modality_reconstructions[0].keys()}
                    adapted_task_modality_reconstruction_losses = torch.stack(adapted_task_modality_reconstruction_losses)

                    return adapted_task_predictions, all_adapted_modality_reconstructions, adapted_task_modality_reconstruction_losses
        # elif self.adaptation_mode == 'MT3_metabatch':
        #     with sdpa_kernel(SDPBackend.MATH):
        #         with torch.enable_grad(): # need to be able to compute gradients even during validation and testing
        #             encoder_parameters = {name: parameter for name, parameter in self.encoder.named_parameters()}
        #             task_modality_decoder_parameters = {name: parameter for name, parameter in self.task_modality_decoder.named_parameters()}

        #             all_iteration_predictions = []
        #             all_adapted_modality_reconstructions = []
        #             all_adapted_task_modality_reconstruction_losses = []

        #             for i in range(len(list(input_data.values())[0])): # for each mini batch
        #                 mini_batch_input_data = {key: value[i].unsqueeze(0) for key, value in input_data.items()}
        #                 mini_batch_task_modality_data = {key: {k: v[i].unsqueeze(0) for k, v in data.items()} for key, data in task_modality_data.items()}
        #                 initial_input_embeddings = functional_call(self.encoder, encoder_parameters, (mini_batch_input_data,))
        #                 initial_task_prediction = self.task_decoder(initial_input_embeddings)
        #                 modality_reconstructions = functional_call(self.task_modality_decoder, task_modality_decoder_parameters, (initial_input_embeddings,))
        #                 task_modality_reconstruction_loss = self.task_modality_decoder_loss(modality_reconstructions, mini_batch_task_modality_data)
        #                 task_modality_reconstruction_loss_encoder_grads = torch.autograd.grad(task_modality_reconstruction_loss, encoder_parameters.values(), create_graph=True) # computes the gradients of the task modality reconstruction loss with respect to the encoder parameters
        #                 task_modality_reconstruction_loss_task_modality_decoder_grads = torch.autograd.grad(task_modality_reconstruction_loss, task_modality_decoder_parameters.values(), create_graph=True) # computes the gradients of the task modality reconstruction loss with respect to the encoder parameters
        #                 adapted_encoder_parameters = {name: param - self.lr * grad for (name, param), grad in zip(encoder_parameters.items(), task_modality_reconstruction_loss_encoder_grads)}
        #                 adapted_task_modality_decoder_parameters = {name: param - self.lr * grad for (name, param), grad in zip(task_modality_decoder_parameters.items(), task_modality_reconstruction_loss_task_modality_decoder_grads)}
        #                 adapted_input_embeddings = functional_call(self.encoder, adapted_encoder_parameters, (mini_batch_input_data,))
        #                 adapted_task_prediction = self.task_decoder(adapted_input_embeddings)
        #                 adapted_modality_reconstructions = functional_call(self.task_modality_decoder, adapted_task_modality_decoder_parameters, (adapted_input_embeddings,))
        #                 adapted_task_modality_reconstruction_loss = self.task_modality_decoder_loss(adapted_modality_reconstructions, mini_batch_task_modality_data)
        #                 iteration_predictions = torch.stack([initial_task_prediction, adapted_task_prediction]) # (num_iterations, batch_size, num_classes, optional height, optional width)

        #                 all_iteration_predictions.append(iteration_predictions)
        #                 all_adapted_modality_reconstructions.append(adapted_modality_reconstructions)
        #                 all_adapted_task_modality_reconstruction_losses.append(adapted_task_modality_reconstruction_loss)

        #             final_iteration_predictions = torch.stack(all_iteration_predictions).squeeze(-1)
        #             final_adapted_modality_reconstructions = {modality: torch.stack([all_adapted_modality_reconstructions[i][modality].squeeze(0) for i in range(len(all_adapted_modality_reconstructions))]).squeeze(-1) for modality in all_adapted_modality_reconstructions[0].keys()}
        #             final_adapted_task_modality_reconstruction_losses = torch.stack(all_adapted_task_modality_reconstruction_losses)

        #             return final_iteration_predictions, final_adapted_modality_reconstructions, final_adapted_task_modality_reconstruction_losses
        # elif self.adaptation_mode == 'sln':
        #     with sdpa_kernel(SDPBackend.MATH):
        #         with torch.enable_grad(): # need to be able to compute gradients even during validation and testing
        #             encoder_parameters = {name: parameter for name, parameter in self.encoder.named_parameters()}
        #             initial_input_embeddings = functional_call(self.encoder, encoder_parameters, (input_data,))
        #             initial_task_prediction = self.task_decoder(initial_input_embeddings)
        #             task_modality_decoder_parameters = {name: parameter for name, parameter in self.task_modality_decoder.named_parameters()}
        #             modality_reconstructions = functional_call(self.task_modality_decoder, task_modality_decoder_parameters, (initial_input_embeddings,))
        #             surrogate_loss = self.surrogate_loss_network(self.modality_reconstruction_loss_calculator(modality_reconstructions, task_modality_data))
        #             surrogate_loss_encoder_grads = torch.autograd.grad(surrogate_loss, encoder_parameters.values(), create_graph=True) # computes the gradients of the task modality reconstruction loss with respect to the encoder parameters
        #             surrogate_loss_task_modality_decoder_grads = torch.autograd.grad(surrogate_loss, task_modality_decoder_parameters.values(), create_graph=True) # computes the gradients of the task modality reconstruction loss with respect to the encoder parameters
        #             adapted_encoder_parameters = {name: param - self.lr * grad for (name, param), grad in zip(encoder_parameters.items(), surrogate_loss_encoder_grads)}
        #             adapted_task_modality_decoder_parameters = {name: param - self.lr * grad for (name, param), grad in zip(task_modality_decoder_parameters.items(), surrogate_loss_task_modality_decoder_grads)}
        #             adapted_input_embeddings = functional_call(self.encoder, adapted_encoder_parameters, (input_data,))
        #             adapted_task_prediction = self.task_decoder(adapted_input_embeddings)
        #             adapted_modality_reconstructions = functional_call(self.task_modality_decoder, adapted_task_modality_decoder_parameters, (adapted_input_embeddings,))
        #             adapted_task_modality_reconstruction_loss = self.task_modality_decoder_loss(adapted_modality_reconstructions, task_modality_data)
        #             iteration_predictions = torch.stack([initial_task_prediction, adapted_task_prediction]) # (num_iterations, batch_size, num_classes, optional height, optional width)

        #             return iteration_predictions, adapted_modality_reconstructions, adapted_task_modality_reconstruction_loss
        # elif self.adaptation_mode == 'sln-10':
        #     iteration_predictions = self.sln(input_data, task_modality_data, num_iterations=10)

        #     return iteration_predictions
        # elif self.adaptation_mode == 'sln-20':
        #     iteration_predictions = self.sln(input_data, task_modality_data, num_iterations=20)

        #     return iteration_predictions
        # elif self.adaptation_mode == 'sln':
        #     self.encoder.eval()
        #     self.task_decoder.eval()
        #     self.task_modality_decoder.eval()

        #     with sdpa_kernel(SDPBackend.MATH):
        #         with torch.enable_grad(): # need to be able to compute gradients even during validation and testing
        #             encoder_parameters = {name: parameter.detach().clone().requires_grad_() for name, parameter in self.encoder.named_parameters()}
        #             initial_input_embeddings = functional_call(self.encoder, encoder_parameters, (input_data,))
        #             # task_decoder_parameters = {name: parameter.detach().clone().requires_grad_() for name, parameter in self.task_decoder.named_parameters()}
        #             initial_task_prediction = functional_call(self.task_decoder, task_decoder_parameters, (initial_input_embeddings,)) if not self.pixelwise else functional_call(self.task_decoder, task_decoder_parameters, (initial_input_embeddings, input_data))
        #             # surrogate_loss = self.surrogate_loss_network(self.modality_reconstruction_loss_calculator(modality_reconstructions=self.task_modality_decoder(initial_input_embeddings, input_data), task_modality_data=task_modality_data), initial_task_prediction)
        #             surrogate_loss = self.surrogate_loss_network(self.modality_reconstruction_loss_calculator(modality_reconstructions=self.task_modality_decoder(initial_input_embeddings, input_data), task_modality_data=task_modality_data))
        #             surrogate_loss_encoder_grads = torch.autograd.grad(surrogate_loss, encoder_parameters.values(), create_graph=True) # computes the gradients of the surrogate loss with respect to the encoder parameters
        #             # surrogate_loss_task_decoder_grads = torch.autograd.grad(surrogate_loss, task_decoder_parameters.values(), create_graph=True) # computes the gradients of the surrogate loss with respect to the encoder parameters
        #             adapted_encoder_parameters = {name: param - self.lr * grad for (name, param), grad in zip(encoder_parameters.items(), surrogate_loss_encoder_grads)}
        #             # adapted_task_decoder_parameters = {name: param - self.lr * grad for (name, param), grad in zip(task_decoder_parameters.items(), surrogate_loss_task_decoder_grads)}
        #             adapted_input_embeddings = functional_call(self.encoder, adapted_encoder_parameters, (input_data,))
        #             adapted_task_prediction = functional_call(self.task_decoder, adapted_task_decoder_parameters, (adapted_input_embeddings,)) if not self.pixelwise else functional_call(self.task_decoder, adapted_task_decoder_parameters, (adapted_input_embeddings, input_data))

        #         iteration_predictions = torch.stack([initial_task_prediction, adapted_task_prediction]) # dim 0 = batch size, dim 1 = num iterations, optional dim 2 = height, optional dim 3 = width

        #         return iteration_predictions
        # elif self.adaptation_mode == 'multimodal_sln':
        #     with sdpa_kernel(SDPBackend.MATH):
        #         with torch.enable_grad(): # need to be able to compute gradients even during validation and testing
        #             encoder_parameters = {name: parameter for name, parameter in self.encoder.named_parameters()}
        #             initial_input_embeddings = functional_call(self.encoder, encoder_parameters, (input_data,))
        #             task_modality_encoder_parameters = {name: parameter for name, parameter in self.task_modality_encoder.named_parameters()}
        #             initial_modality_embeddings = functional_call(self.task_modality_encoder, task_modality_encoder_parameters, (task_modality_data,))
        #             initial_modality_embeddings = self.upsample_modality_embeddings(initial_modality_embeddings, initial_input_embeddings)
        #             initial_concatenated_embeddings = torch.cat([initial_input_embeddings, initial_modality_embeddings], dim=1)
        #             initial_task_prediction = self.task_decoder(initial_concatenated_embeddings)
        #             task_modality_decoder_parameters = {name: parameter for name, parameter in self.task_modality_decoder.named_parameters()}
        #             modality_reconstructions = functional_call(self.task_modality_decoder, task_modality_decoder_parameters, (initial_concatenated_embeddings,))
        #             surrogate_loss = self.surrogate_loss_network(initial_concatenated_embeddings, self.modality_reconstruction_loss_calculator(modality_reconstructions, task_modality_data))
        #             surrogate_loss_encoder_grads = torch.autograd.grad(surrogate_loss, encoder_parameters.values(), create_graph=True) # computes the gradients of the surrogate loss with respect to the encoder parameters
        #             surrogate_loss_task_modality_encoder_grads = torch.autograd.grad(surrogate_loss, task_modality_encoder_parameters.values(), create_graph=True) # computes the gradients of the surrogate loss with respect to the encoder parameters
        #             surrogate_loss_task_modality_decoder_grads = torch.autograd.grad(surrogate_loss, task_modality_decoder_parameters.values(), create_graph=True) # computes the gradients of the surrogate loss with respect to the encoder parameters
        #             adapted_encoder_parameters = {name: param - self.lr * grad for (name, param), grad in zip(encoder_parameters.items(), surrogate_loss_encoder_grads)}
        #             adapted_task_modality_encoder_parameters = {name: param - self.lr * grad for (name, param), grad in zip(task_modality_encoder_parameters.items(), surrogate_loss_task_modality_encoder_grads)}
        #             adapted_task_modality_decoder_parameters = {name: param - self.lr * grad for (name, param), grad in zip(task_modality_decoder_parameters.items(), surrogate_loss_task_modality_decoder_grads)}
        #             adapted_input_embeddings = functional_call(self.encoder, adapted_encoder_parameters, (input_data,))
        #             adapted_modality_embeddings = functional_call(self.task_modality_encoder, task_modality_encoder_parameters, (task_modality_data,))
        #             adapted_modality_embeddings = self.upsample_modality_embeddings(adapted_modality_embeddings, adapted_input_embeddings)
        #             adapted_concatenated_embeddings = torch.cat([adapted_input_embeddings, adapted_modality_embeddings], dim=1)
        #             adapted_task_prediction = self.task_decoder(adapted_concatenated_embeddings)
        #             adapted_modality_reconstructions = functional_call(self.task_modality_decoder, adapted_task_modality_decoder_parameters, (adapted_concatenated_embeddings,))
        #             adapted_task_modality_reconstruction_loss = self.task_modality_decoder_loss(adapted_modality_reconstructions, task_modality_data)
        #             iteration_predictions = torch.stack([initial_task_prediction, adapted_task_prediction]) # (num_iterations, batch_size, num_classes, optional height, optional width)

        #             return iteration_predictions, adapted_modality_reconstructions, adapted_task_modality_reconstruction_loss
        # elif self.adaptation_mode == 'sln_input_embeddings':
        #     self.encoder.eval()
        #     self.task_decoder.eval()

        #     with sdpa_kernel(SDPBackend.MATH):
        #         with torch.enable_grad(): # need to be able to compute gradients even during validation and testing
        #             encoder_parameters = {name: parameter.detach().clone().requires_grad_() for name, parameter in self.encoder.named_parameters()}
        #             initial_input_embeddings = functional_call(self.encoder, encoder_parameters, (input_data,))
        #             task_decoder_parameters = {name: parameter.detach().clone().requires_grad_() for name, parameter in self.task_decoder.named_parameters()}
        #             initial_task_prediction = functional_call(self.task_decoder, task_decoder_parameters, (initial_input_embeddings,)) if not self.pixelwise else functional_call(self.task_decoder, task_decoder_parameters, (initial_input_embeddings, input_data))
        #             surrogate_loss = self.surrogate_loss_network(initial_input_embeddings, initial_task_prediction)
        #             surrogate_loss_encoder_grads = torch.autograd.grad(surrogate_loss, encoder_parameters.values(), create_graph=True) # computes the gradients of the surrogate loss with respect to the encoder parameters
        #             surrogate_loss_task_decoder_grads = torch.autograd.grad(surrogate_loss, task_decoder_parameters.values(), create_graph=True) # computes the gradients of the surrogate loss with respect to the task decoder parameters
        #             adapted_encoder_parameters = {name: param - self.lr * grad for (name, param), grad in zip(encoder_parameters.items(), surrogate_loss_encoder_grads)}
        #             adapted_task_decoder_parameters = {name: param - self.lr * grad for (name, param), grad in zip(task_decoder_parameters.items(), surrogate_loss_task_decoder_grads)}
        #             adapted_input_embeddings = functional_call(self.encoder, adapted_encoder_parameters, (input_data,))
        #             adapted_task_prediction = functional_call(self.task_decoder, adapted_task_decoder_parameters, (adapted_input_embeddings,)) if not self.pixelwise else functional_call(self.task_decoder, adapted_task_decoder_parameters, (adapted_input_embeddings, input_data))

        #         iteration_predictions = torch.cat([initial_task_prediction, adapted_task_prediction], dim=1) # dim 0 = batch size, dim 1 = num iterations, optional dim 2 = height, optional dim 3 = width

        #         if len(iteration_predictions.shape) == 2:
        #             iteration_predictions = iteration_predictions.t().unsqueeze(-1) # (num_iterations, batch_size, 1)
        #         else:
        #             iteration_predictions = iteration_predictions.permute(1,0,2,3).unsqueeze(2) # (num_iterations, batch_size, 1, height, width)

        #         return iteration_predictions
        # elif self.adaptation_mode == 'sln_encode':
        #     self.encoder.eval()
        #     self.task_modality_encoder.eval()
        #     self.task_decoder.eval()

        #     with sdpa_kernel(SDPBackend.MATH):
        #         with torch.enable_grad(): # need to be able to compute gradients even during validation and testing
        #             encoder_parameters = {name: parameter.detach().clone().requires_grad_() for name, parameter in self.encoder.named_parameters()}
        #             initial_input_embeddings = functional_call(self.encoder, encoder_parameters, (input_data,))
        #             task_modality_encoder_parameters = {name: parameter.detach().clone().requires_grad_() for name, parameter in self.task_modality_encoder.named_parameters()}
        #             initial_modality_embeddings = functional_call(self.task_modality_encoder, task_modality_encoder_parameters, (task_modality_data,))

        #             if initial_modality_embeddings.shape[2] != initial_input_embeddings.shape[2]:
        #                 initial_modality_embeddings = nn.Upsample(size=initial_input_embeddings.shape[2], mode='bilinear')(initial_modality_embeddings) # upsamples bilinearly to match the spatial dimensions of the input embeddings

        #             concatenated_embeddings = torch.cat([initial_input_embeddings, initial_modality_embeddings], dim=1)
        #             task_decoder_parameters = {name: parameter.detach().clone().requires_grad_() for name, parameter in self.task_decoder.named_parameters()}
        #             initial_task_prediction = functional_call(self.task_decoder, task_decoder_parameters, (concatenated_embeddings,)) if not self.pixelwise else functional_call(self.task_decoder, task_decoder_parameters, (concatenated_embeddings, input_data))
        #             surrogate_loss = self.surrogate_loss_network(concatenated_embeddings, initial_task_prediction)
        #             surrogate_loss_encoder_grads = torch.autograd.grad(surrogate_loss, encoder_parameters.values(), create_graph=True) # computes the gradients of the surrogate loss with respect to the encoder parameters
        #             surrogate_loss_task_decoder_grads = torch.autograd.grad(surrogate_loss, task_decoder_parameters.values(), create_graph=True) # computes the gradients of the surrogate loss with respect to the encoder parameters
        #             surrogate_loss_modality_encoder_grads = torch.autograd.grad(surrogate_loss, task_modality_encoder_parameters.values(), create_graph=True) # computes the gradients of the surrogate loss with respect to the encoder parameters

        #             adapted_encoder_parameters = {name: param - self.lr * grad for (name, param), grad in zip(encoder_parameters.items(), surrogate_loss_encoder_grads)}
        #             adapted_task_modality_encoder_parameters = {name: param - self.lr * grad for (name, param), grad in zip(task_modality_encoder_parameters.items(), surrogate_loss_modality_encoder_grads)}
        #             adapted_task_decoder_parameters = {name: param - self.lr * grad for (name, param), grad in zip(task_decoder_parameters.items(), surrogate_loss_task_decoder_grads)}
        #             adapted_input_embeddings = functional_call(self.encoder, adapted_encoder_parameters, (input_data,))
        #             adapted_modality_embeddings = functional_call(self.task_modality_encoder, adapted_task_modality_encoder_parameters, (task_modality_data,))

        #             if adapted_modality_embeddings.shape[2] != adapted_input_embeddings.shape[2]:
        #                 adapted_modality_embeddings = nn.Upsample(size=adapted_input_embeddings.shape[2], mode='bilinear')(adapted_modality_embeddings) # upsamples bilinearly to match the spatial dimensions of the input embeddings

        #             adapted_concatenated_embeddings = torch.cat([adapted_input_embeddings, adapted_modality_embeddings], dim=1)
        #             adapted_task_prediction = functional_call(self.task_decoder, adapted_task_decoder_parameters, (adapted_concatenated_embeddings,)) if not self.pixelwise else functional_call(self.task_decoder, adapted_task_decoder_parameters, (adapted_concatenated_embeddings, input_data))

        #         iteration_predictions = torch.cat([initial_task_prediction, adapted_task_prediction], dim=1) # dim 0 = batch size, dim 1 = num iterations, optional dim 2 = height, optional dim 3 = width

        #         if len(iteration_predictions.shape) == 2:
        #             iteration_predictions = iteration_predictions.t().unsqueeze(-1) # (num_iterations, batch_size, 1)
        #         else:
        #             iteration_predictions = iteration_predictions.permute(1,0,2,3).unsqueeze(2) # (num_iterations, batch_size, 1, height, width)

        #         return iteration_predictions
        # elif self.adaptation_mode == 'rna':
        #     self.encoder.eval()
        #     self.task_decoder.eval()
        #     self.task_modality_decoder.eval()

        #     initial_input_embeddings = self.encoder(input_data)
        #     initial_task_prediction = self.task_decoder(initial_input_embeddings) if not self.pixelwise else self.task_decoder(initial_input_embeddings, input_data)
        #     adapted_input_embeddings = self.feature_adapter(initial_input_embeddings=initial_input_embeddings, feature_adapter_parameters=self.hypernetwork(modality_reconstruction_losses=self.modality_reconstruction_loss_calculator(modality_reconstructions=self.task_modality_decoder(initial_input_embeddings, input_data), task_modality_data=task_modality_data), initial_task_prediction=initial_task_prediction))
        #     adapted_task_prediction = self.task_decoder(adapted_input_embeddings) if not self.pixelwise else self.task_decoder(adapted_input_embeddings, input_data)

        #     iteration_predictions = torch.cat([initial_task_prediction, adapted_task_prediction], dim=1) # dim 0 = batch size, dim 1 = num iterations, optional dim 2 = height, optional dim 3 = width

        #     if len(iteration_predictions.shape) == 2:
        #         iteration_predictions = iteration_predictions.t().unsqueeze(-1) # (num_iterations, batch_size, 1)
        #     else:
        #         iteration_predictions = iteration_predictions.permute(1,0,2,3).unsqueeze(2) # (num_iterations, batch_size, 1, height, width)

        #     return iteration_predictions
        # elif self.adaptation_mode == 'rna_input_embeddings':
        #     self.encoder.eval()
        #     self.task_decoder.eval()

        #     initial_input_embeddings = self.encoder(input_data)
        #     initial_task_prediction = self.task_decoder(initial_input_embeddings) if not self.pixelwise else self.task_decoder(initial_input_embeddings, input_data)
        #     adapted_input_embeddings = self.feature_adapter(initial_input_embeddings=initial_input_embeddings, feature_adapter_parameters=self.hypernetwork(initial_input_embeddings, initial_task_prediction))
        #     adapted_task_prediction = self.task_decoder(adapted_input_embeddings) if not self.pixelwise else self.task_decoder(adapted_input_embeddings, input_data)

        #     iteration_predictions = torch.cat([initial_task_prediction, adapted_task_prediction], dim=1) # dim 0 = batch size, dim 1 = num iterations, optional dim 2 = height, optional dim 3 = width

        #     if len(iteration_predictions.shape) == 2:
        #         iteration_predictions = iteration_predictions.t().unsqueeze(-1) # (num_iterations, batch_size, 1)
        #     else:
        #         iteration_predictions = iteration_predictions.permute(1,0,2,3).unsqueeze(2) # (num_iterations, batch_size, 1, height, width)

        #     return iteration_predictions
        # elif self.adaptation_mode == 'MT3_frozen':
        #     # keep batch norm and dropout deterministic
        #     self.encoder.eval()
        #     self.task_decoder.eval()

        #     with torch.enable_grad(): # need to be able to compute gradients even during validation and testing
        #         encoder_parameters = {name: parameter.detach().clone().requires_grad_() for name, parameter in self.encoder.named_parameters()}
        #         initial_input_embeddings = functional_call(self.encoder, encoder_parameters, (input_data,))
        #         initial_task_prediction = self.task_decoder(initial_input_embeddings)
        #         task_modality_reconstruction_loss = self.task_modality_decoder_loss(modality_reconstructions=self.task_modality_decoder(initial_input_embeddings, task_modality_data), task_modality_data=task_modality_data)
        #         task_modality_reconstruction_loss_grads = torch.autograd.grad(task_modality_reconstruction_loss, encoder_parameters.values(), create_graph=True) # computes the gradients of the task modality reconstruction loss with respect to the encoder parameters
        #         adapted_encoder_parameters = {name: param - self.lr * grad for (name, param), grad in zip(encoder_parameters.items(), task_modality_reconstruction_loss_grads)}
        #         adapted_input_embeddings = functional_call(self.encoder, adapted_encoder_parameters, (task_modality_data,))
        #         adapted_task_prediction = self.task_decoder(adapted_input_embeddings)

        #     return torch.cat([initial_task_prediction, adapted_task_prediction], dim=1).t().unsqueeze(-1) # (num_iterations+1, batch_size, 1)

class Model(LightningModule):
    def __init__(self, task, architecture, adaptation_mode, pretrained, max_lr, weight_decay, warmup_epochs, num_train_batches, min_lr, epochs, inner_loop_lr):
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

    def configure_models(self):
        pixelwise = self.hparams.task == 'biomass'
        self.num_classes = 100 if self.hparams.task == 'species' else 1
        self.model = EncoderDecoder(task=self.hparams.task,
                                    architecture=self.hparams.architecture,
                                    adaptation_mode=self.hparams.adaptation_mode,
                                    pixelwise=pixelwise,
                                    num_classes=self.num_classes,
                                    pretrained=self.hparams.pretrained,
                                    lr=self.hparams.inner_loop_lr)

    def configure_metrics(self):
        if self.hparams.adaptation_mode == 'TMD':
            metric_collection = {}
        else:
            if self.hparams.task == 'species':
                metric_collection = {'MAP': MultilabelAveragePrecision(self.num_classes), 'Recall': MultilabelRecall(self.num_classes)}
            else:
                metric_collection = {'R2': R2Score(), 'RMSE': MeanSquaredError(squared=False)}

        if self.hparams.adaptation_mode in ['TMD', 'JT', 'MT3_metabatch']:
            for modality in task_modalities:
                if modality in categorical_modalities:
                    metric_collection[f'{modality} accuracy'] = MulticlassAccuracy(num_classes=no_data_values[modality], ignore_index=no_data_values[modality])
                else:
                    metric_collection[f'{modality} R2'] = R2Score()

        for split in ['train', 'val', 'random_test', 'geographic_test']:
            if split == 'train' and self.hparams.adaptation_mode == 'UDA-SS':
                setattr(self, f'{split}_metrics', MetricCollection({**metric_collection, 'DED': DomainEmbeddingDistance(self.model.embedding_dim)}).clone(prefix=f'{split.replace("_", " ").capitalize()} '))
            else:
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
        # elif self.hparams.adaptation_mode == 'UDA-SS':
        #     embeddings, prediction, modality_reconstructions, task_modality_reconstruction_loss = prediction
        # elif 'ttt' in self.hparams.adaptation_mode or 'TTT' in self.hparams.adaptation_mode or '-10' in self.hparams.adaptation_mode or '-20' in self.hparams.adaptation_mode or self.hparams.adaptation_mode in ['sln_input_embeddings', 'sln_encode', 'rna', 'rna_input_embeddings']:
        #     iteration_predictions = prediction
        #     prediction = iteration_predictions[-1]
        # elif self.hparams.adaptation_mode in ['MT3', 'multimodal_MT3', 'sln', 'multimodal_sln']:
        #     iteration_predictions, modality_reconstructions, task_modality_reconstruction_loss = prediction
        #     prediction = iteration_predictions[-1]
        # elif self.hparams.adaptation_mode == 'MT3_metabatch':
            # iteration_predictions, modality_reconstructions, task_modality_reconstruction_loss = prediction
            # prediction = iteration_predictions[:, -1]
            # prediction, modality_reconstructions, task_modality_reconstruction_loss = prediction

        #     print(iteration_predictions.shape, task_modality_reconstruction_loss.shape)
        # elif self.hparams.adaptation_mode == 'MT3_frozen' or self.hparams.adaptation_mode == 'rna':
        #     iteration_predictions = prediction
        #     prediction = iteration_predictions[-1]

        # if self.hparams.adaptation_mode == 'UDA-SS':
        #     if mode == 'train':
        #         # filter predictions and targets to only include labeled source samples
        #         labeled_source_mask = torch.tensor([d == 'labeled_source' for d in domain]) # creates mask for labeled source samples
        #         prediction = prediction[labeled_source_mask]
        #         target = target[labeled_source_mask]

        if self.hparams.adaptation_mode != 'TMD' and self.hparams.task == 'biomass':
            valid_mask = target != biomass_no_data_value # mask for the NaN pixels in the target
            target = target[valid_mask]

            # if 'ttt' in self.hparams.adaptation_mode or 'TTT' in self.hparams.adaptation_mode or '-10' in self.hparams.adaptation_mode or '-20' in self.hparams.adaptation_mode or self.hparams.adaptation_mode in ['MT3', 'multimodal_MT3', 'multimodal_sln', 'sln', 'sln_input_embeddings', 'sln_encode', 'rna', 'rna_input_embeddings']:
            if 'TTT' in self.hparams.adaptation_mode and mode == 'val':
                iteration_predictions = torch.stack([pred[valid_mask] for pred in iteration_predictions])
            else:
                prediction = prediction[valid_mask]

        # LOSS #

        if self.hparams.adaptation_mode in ['JT', 'UDA-SS', 'multimodal_joint_training', 'joint_probing', 'MT3', 'sln', 'multimodal_MT3', 'MT3_metabatch', 'multimodal_sln', 'TMD']:
            with torch.no_grad():
                modality_reconstruction_losses = ModalityReconstructionLossCalculator()(modality_reconstructions, task_modality_data)

                for modality, loss in modality_reconstruction_losses.items():
                    mean_loss = loss.nanmean()

                    if not mean_loss.isnan():
                        self.log(f'{mode.capitalize().replace("_", " ")} {modality} reconstruction loss', mean_loss, add_dataloader_idx=False)

        if self.hparams.adaptation_mode == 'TMD':
            target = task_modality_data # task is to reconstruct the modalities

        # if self.hparams.adaptation_mode == 'MT3_metabatch':
        #     task_losses = torch.stack([self.criterion(prediction[i], target[i]) for i in range(prediction.shape[0])])
        #     loss = (task_losses + task_modality_reconstruction_loss).mean()
        if 'TTT' in self.hparams.adaptation_mode and mode == 'val':
            iteration_losses = np.array([self.criterion(pred, target).item() for pred in iteration_predictions])
            best_iteration_number = np.argmin(iteration_losses) + 1
            # self.model.val_best_num_iterations = min(self.model.val_best_num_iterations, np.argmin(iteration_losses)+1)
            # prediction = iteration_predictions[self.model.val_best_num_iterations-1]
            print(f'Best TTT iteration: {best_iteration_number}')
            self.val_batches_best_num_iterations.append(best_iteration_number)
            prediction = iteration_predictions[best_iteration_number-1]

        loss = self.criterion(prediction, target) # computes the loss

        if self.hparams.adaptation_mode in ['JT', 'MT3_metabatch', 'UDA-SS', 'multimodal_joint_training', 'joint_probing', 'MT3', 'sln', 'multimodal_MT3', 'multimodal_sln']:
            self.log(f'{mode.capitalize().replace("_", " ")} task modality reconstruction loss', task_modality_reconstruction_loss, add_dataloader_idx=False) # logs the task modality reconstruction loss
            loss += task_modality_reconstruction_loss

        self.log(f'{mode.capitalize().replace("_", " ")} loss', loss, add_dataloader_idx=False) # logs the loss

        # METRICS #

        # log adaptation improvement over iterations
        # if self.hparams.adaptation_mode in ['ttt-jt', 'ttt-mjt', 'ttt-jp', 'ttt-adapter', 'MT3', 'multimodal_MT3', 'multimodal_sln', 'sln', 'sln_input_embeddings', 'sln_encode', 'MT3_frozen', 'rna', 'rna_input_embeddings']:
        #     metric = MultilabelAveragePrecision(self.num_classes, reduction='none') if self.hparams.task == 'species' else R2Score(reduction='none')
        #     # losses = torch.stack([nn.MSELoss(reduction='none')(prediction, target) if self.hparams.task != 'species' else nn.BCEWithLogitsLoss(reduction='none')(prediction, target) for prediction in iteration_predictions]) # (num iterations+1, batch size or num pixels, optional 1)
        #     values = torch.stack([metric(prediction, target) for prediction in iteration_predictions])

        #     if len(losses.shape) > 2:
        #         losses = losses.squeeze(-1) # removes the last dimension if it is 1

        #     if self.hparams.task == 'species':
        #         losses = losses.mean(dim=-1) # averages over the classes

        #     # if not hasattr(self, f'{mode}_losses'):
        #     #     setattr(self, f'{mode}_losses', [])

        #     # getattr(self, f'{mode}_losses').append(losses.detach().cpu())
        #     self.__dict__.setdefault(f'{mode}_losses', []).append(losses.detach().cpu())

        #     # retrieve the number of batches in the current dataloader
        #     if mode == 'train':
        #         num_batches = self.trainer.num_training_batches
        #     elif mode == 'val':
        #         num_batches = self.trainer.num_val_batches[0]
        #     else:
        #         num_batches = self.trainer.num_test_batches[dataloader_idx]

        #     if batch_idx == num_batches-1: # if we are on the last batch
        #         losses = torch.cat(getattr(self, f'{mode}_losses'), dim=1).mean(dim=1) # length is the number of iterations
        #         initial_loss = losses[0].item()
        #         final_loss = losses[-1].item()
        #         adaptation_improvement = (initial_loss - final_loss) / initial_loss * 100

        #         self.log(f'{mode.capitalize().replace("_", " ")} adaptation improvement %', adaptation_improvement, add_dataloader_idx=False)
        #         self._plot_loss_over_iterations(mode)

        #         setattr(self, f'{mode}_losses', []) # resets for the next epoch

        if self.hparams.task == 'species':
            prediction = torch.sigmoid(prediction) # converts logits to probabilities
            target = target.long()

        metrics = getattr(self, f'{mode}_metrics')

        for name, metric in metrics.items():
            # if 'DED' in name:
            #     metric.update(embeddings, domain) # updates the DED metric for this batch
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

        if self.hparams.adaptation_mode in ['TMD', 'JT'] and self.trainer.is_last_batch:
            task_modality_reconstruction_performance = torch.stack([metric.compute() for name, metric in metrics.items() if 'accuracy' in name or 'R2' in name]).nanmean() # computes the mean performance across all modalities and tiles, ignoring NaNs
            self.log(f'{mode.capitalize().replace("_", " ")} task modality reconstruction performance', task_modality_reconstruction_performance, on_step=False, on_epoch=True, add_dataloader_idx=False) # logs the task modality reconstruction performance

        # log the images in the first batch
        if batch_idx == 0: # if we are on the first batch
            self._log_images(task_modality_data['Sentinel2']['data'].cpu().numpy()[:, [3,2,1]].astype(float), mode)

        if mode == 'train':
            return loss

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch=batch, batch_idx=batch_idx, mode='train')

        return loss

    def validation_step(self, batch, batch_idx):
        self.general_step(batch=batch, batch_idx=batch_idx, mode='val')

    def test_step(self, batch, batch_idx, dataloader_idx):
        # if 'uda' in self.hparams.adaptation_mode and batch_idx == 0: # if we are in UDA mode and on the first batch
        #     if dataloader_idx == 0: # if we are on the first dataloader
        #         self.original_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()} # saves the original model state
        #     else:
        #         self.model.load_state_dict(self.original_state_dict) # loads the original model state

        #     self._perform_uda_training(dataloader_idx)
        if 'TTT' in self.hparams.adaptation_mode:
            self.model.mode = 'test'

        self.general_step(batch=batch, batch_idx=batch_idx, mode='random_test' if dataloader_idx==0 else 'geographic_test', dataloader_idx=dataloader_idx)

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch to determine best iteration for TTT"""
        if 'TTT' in self.hparams.adaptation_mode:
            # Calculate average best iteration across all batches
            self.model.val_best_num_iterations = int(round(np.mean(self.val_batches_best_num_iterations)))
            # self.model.val_best_num_iterations = int(round(np.median(self.val_batches_best_num_iterations)))
            print(f'Val best num TTT iterations: {self.model.val_best_num_iterations}')

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

    def _perform_uda_training(self, dataloader_idx):
        """
        Perform unsupervised domain adaptation on unlabeled test data for the specified split.
        This trains the encoder and task modality decoder jointly on the test data.
        """

        dataloader = self.trainer.datamodule.test_dataloader()[dataloader_idx]
        split_name = 'random_test' if dataloader_idx == 0 else 'geographic_test'

        print(f'Starting UDA training on {split_name.replace("_", " ")} split...')

        # set model components to training mode
        # self.model.encoder.train()
        self.model.task_modality_decoder.train()

        # uda_parameters = list(self.model.encoder.parameters()) + list(self.model.task_modality_decoder.parameters())
        uda_parameters = list(self.model.task_modality_decoder.parameters())

        # ensure all parameters require gradients
        for param in uda_parameters:
            param.requires_grad = True

        optimizer = torch.optim.AdamW(uda_parameters, lr=self.hparams.max_lr, weight_decay=self.hparams.weight_decay)

        # UDA training loop
        num_uda_epochs = 50
        device = self.device

        for epoch in range(num_uda_epochs):
            epoch_loss = 0.0
            num_batches = 0

            with torch.enable_grad():
                for _, batch in enumerate(dataloader):
                    input_data, task_modality_data, _ = batch # ignores targets for UDA
                    input_data = {modality: data.to(device) for modality, data in input_data.items()} # moves input data to GPU
                    task_modality_data = {modality: {key: value.to(device) for key, value in dictionary.items()} for modality, dictionary in task_modality_data.items()} # moves task modality data to GPU
                    optimizer.zero_grad() # zeroes the gradients
                    input_embeddings = self.model.encoder(input_data) # embeds the input data
                    modality_reconstructions = self.model.task_modality_decoder(input_embeddings) # reconstructs the task modalities
                    task_modality_reconstruction_loss = self.model.task_modality_decoder_loss(modality_reconstructions, task_modality_data) # computes the task modality reconstruction loss
                    task_modality_reconstruction_loss.backward() # computes the gradients of the task modality reconstruction loss with respect to the encoder and task modality decoder parameters
                    optimizer.step() # updates the encoder and task modality decoder parameters
                    optimizer.zero_grad() # zeroes the gradients
                    epoch_loss += task_modality_reconstruction_loss.item()
                    num_batches += 1

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            print(f'UDA Epoch {epoch+1}/{num_uda_epochs} on {split_name}: Loss = {avg_loss:.4f}')

        print(f'UDA training completed on {split_name} split')
