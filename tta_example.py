# imports
from lightning.pytorch import LightningModule
from torch.autograd import grad
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

# global variables
batch_size = 4
image_size = 16
embedding_dim = 32
epochs = 5
lr = 1e-3

# data
modalities = {'sentinel2': torch.randn(batch_size, 12, image_size, image_size),
              'sentinel1': torch.randn(batch_size, 8, image_size, image_size),
              'asterdem': torch.randn(batch_size, 2, image_size, image_size),
              'ethgch': torch.randn(batch_size, 2, image_size, image_size)}

downstream_task = torch.randn(batch_size, 1, image_size, image_size)

# models
class PretrainedEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Conv2d(in_channels=20, out_channels=embedding_dim, kernel_size=8, stride=8)

    def forward(self, encoder_input_modalities):
        return self.model(encoder_input_modalities)

class TaskDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(nn.Upsample(size=(image_size, image_size), mode='bilinear'),
                                   nn.Conv2d(in_channels=embedding_dim, out_channels=1, kernel_size=1))

    def forward(self, input_embeddings):
        return self.model(input_embeddings)

class TaskModalityDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(nn.Upsample(size=(image_size, image_size), mode='bilinear'),
                                   nn.Conv2d(in_channels=embedding_dim, out_channels=24, kernel_size=1))

    def forward(self, input_embeddings):
        modality_reconstructions_raw = self.model(input_embeddings)
        modality_reconstructions = {'sentinel2': modality_reconstructions_raw[:, :12],
                                    'sentinel1': modality_reconstructions_raw[:, 12:20],
                                    'asterdem': modality_reconstructions_raw[:, 20:22],
                                    'ethgch': modality_reconstructions_raw[:, 22:24]}

        return modality_reconstructions

class SurrogateLossMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.surrogate_loss = nn.Sequential(nn.Linear(4, 32),
                                            nn.ReLU(),
                                            nn.Linear(32, 16),
                                            nn.ReLU(),
                                            nn.Linear(16, 1))

    def forward(self, modality_reconstruction_losses):
        modality_reconstruction_losses = torch.stack(list(modality_reconstruction_losses.values()))
        surrogate_loss = self.surrogate_loss(modality_reconstruction_losses)

        return surrogate_loss

# functions
def get_model_state(model):
    return {name: param.clone() for name, param in model.named_parameters()}

def check_model_update(old_model, new_model):
    for name in old_model:
        if not torch.equal(old_model[name], new_model[name]):
            return True
    return False

# modules
pretrained_encoder = PretrainedEncoder()
task_modality_decoder = TaskModalityDecoder()
surrogate_loss_mlp = SurrogateLossMLP()
task_decoder = TaskDecoder()
mse_loss = nn.MSELoss()

# optimizers
optimizer_pretrained_encoder = optim.Adam(pretrained_encoder.parameters(), lr=lr)
optimizer_task_modality_decoder = optim.Adam(task_modality_decoder.parameters(), lr=lr)
optimizer_surrogate_loss_mlp = optim.Adam(surrogate_loss_mlp.parameters(), lr=lr)
optimizer_task_decoder = optim.Adam(task_decoder.parameters(), lr=lr)

encoder_input_modalities = torch.cat((modalities['sentinel2'], modalities['sentinel1']), dim=1)

print(f'Pretrained encoder: {sum(p.numel() for p in pretrained_encoder.parameters())} parameters, {sum(p.numel() for p in pretrained_encoder.parameters() if p.requires_grad)} trainable parameters')
print(f'Task modality decoder: {sum(p.numel() for p in task_modality_decoder.parameters())} parameters, {sum(p.numel() for p in task_modality_decoder.parameters() if p.requires_grad)} trainable parameters')
print(f'Surrogate loss MLP: {sum(p.numel() for p in surrogate_loss_mlp.parameters())} parameters, {sum(p.numel() for p in surrogate_loss_mlp.parameters() if p.requires_grad)} trainable parameters')
print(f'Task decoder: {sum(p.numel() for p in task_decoder.parameters())} parameters, {sum(p.numel() for p in task_decoder.parameters() if p.requires_grad)} trainable parameters')

for i in range(epochs):
    print(f'Epoch {i+1}/{epochs}')

    # model states at step 0
    pretrained_encoder_state_0 = get_model_state(pretrained_encoder)
    task_modality_decoder_state_0 = get_model_state(task_modality_decoder)
    surrogate_loss_mlp_state_0 = get_model_state(surrogate_loss_mlp)
    task_decoder_state_0 = get_model_state(task_decoder)

    # clear gradients
    optimizer_surrogate_loss_mlp.zero_grad()
    optimizer_task_modality_decoder.zero_grad()
    optimizer_pretrained_encoder.zero_grad()
    optimizer_task_decoder.zero_grad()

    # forward pass 1
    input_embeddings = pretrained_encoder(encoder_input_modalities)
    modality_reconstructions = task_modality_decoder(input_embeddings)
    reconstruction_losses = {modality: mse_loss(reconstruction, modalities[modality]) for modality, reconstruction in modality_reconstructions.items()}
    surrogate_loss = surrogate_loss_mlp(reconstruction_losses)

    # backward pass 1
    surrogate_loss.backward(create_graph=True)

    # update parameters
    optimizer_task_modality_decoder.step()
    optimizer_pretrained_encoder.step()

    # pretrained_encoder_grads = grad(surrogate_loss, pretrained_encoder.parameters(), create_graph=True, retrain_graph=True)
    # finetuned_encoder_parameters = [param - lr * grad_param for param, grad_param in zip(pretrained_encoder.parameters(), pretrained_encoder_grads)]

    # model states after step 1
    pretrained_encoder_state_1 = get_model_state(pretrained_encoder)
    task_modality_decoder_state_1 = get_model_state(task_modality_decoder)
    surrogate_loss_mlp_state_1 = get_model_state(surrogate_loss_mlp)
    task_decoder_state_1 = get_model_state(task_decoder)

    # check if models were updated
    print('After backpropagating the surrogate loss')
    print(f'Pretrained encoder updated: {check_model_update(pretrained_encoder_state_0, pretrained_encoder_state_1)}')
    print(f'Task modality decoder updated: {check_model_update(task_modality_decoder_state_0, task_modality_decoder_state_1)}')
    print(f'Surrogate loss MLP updated: {check_model_update(surrogate_loss_mlp_state_0, surrogate_loss_mlp_state_1)}')
    print(f'Task decoder updated: {check_model_update(task_decoder_state_0, task_decoder_state_1)}')

    # forward pass 2
    updated_input_embeddings = pretrained_encoder(encoder_input_modalities)
    assert not torch.equal(input_embeddings, updated_input_embeddings)
    downstream_task_prediction = task_decoder(updated_input_embeddings)
    task_loss = mse_loss(downstream_task_prediction, downstream_task)

    # optimizer_task_decoder.zero_grad()
    # optimizer_pretrained_encoder.zero_grad()
    # optimizer_task_modality_decoder.zero_grad()
    # optimizer_surrogate_loss_mlp.zero_grad()

    # backward pass 2
    task_loss.backward()

    # update parameters
    optimizer_surrogate_loss_mlp.step()
    optimizer_task_decoder.step()
    optimizer_pretrained_encoder.step()

    # model states after step 2
    pretrained_encoder_state_2 = get_model_state(pretrained_encoder)
    task_modality_decoder_state_2 = get_model_state(task_modality_decoder)
    surrogate_loss_mlp_state_2 = get_model_state(surrogate_loss_mlp)
    task_decoder_state_2 = get_model_state(task_decoder)

    # check if models were updated
    print('After backpropagating the task loss')
    print(f'Pretrained encoder updated: {check_model_update(pretrained_encoder_state_1, pretrained_encoder_state_2)}')
    print(f'Task modality decoder updated: {check_model_update(task_modality_decoder_state_1, task_modality_decoder_state_2)}')
    print(f'Surrogate loss MLP updated: {check_model_update(surrogate_loss_mlp_state_1, surrogate_loss_mlp_state_2)}')
    print(f'Task decoder updated: {check_model_update(task_decoder_state_1, task_decoder_state_2)}')

    print(f'Task loss: {task_loss.item():.4f}')
    exit()

# encoder_input_modalities = torch.cat((modalities['sentinel2'], modalities['sentinel1']), dim=1)
# print(f'Encoder input modalities shape: {encoder_input_modalities.shape}')
# input_embeddings = pretrained_encoder(encoder_input_modalities)
# print(f'Input embeddings shape: {input_embeddings.shape}')
# downstream_task_prediction = task_decoder(input_embeddings)
# print(f'Downstream task prediction shape: {downstream_task_prediction.shape}')
# modality_reconstructions_raw = task_modality_decoder(input_embeddings)
# modality_reconstructions = {'sentinel2': modality_reconstructions_raw[:, :12],
#                             'sentinel1': modality_reconstructions_raw[:, 12:20],
#                             'asterdem': modality_reconstructions_raw[:, 20:22],
#                             'ethgch': modality_reconstructions_raw[:, 22:24]}
# print(f'Sentinel-2 reconstruction shape: {modality_reconstructions["sentinel2"].shape}')
# print(f'Sentinel-1 reconstruction shape: {modality_reconstructions["sentinel1"].shape}')
# print(f'AsterDEM reconstruction shape: {modality_reconstructions["asterdem"].shape}')
# print(f'ETHGCH reconstruction shape: {modality_reconstructions["ethgch"].shape}')
# reconstruction_losses = {modality: mse_loss(reconstruction, globals()[modality]) for modality, reconstruction in modality_reconstructions.items()}
# surrogate_loss = surrogate_loss_mlp(reconstruction_losses)
# print(f'Surrogate loss value: {surrogate_loss.item()}')

# class TTAModel(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.pretrained_encoder = PretrainedEncoder()
#         self.task_decoder = TaskDecoder()
#         self.task_modality_decoder = TaskModalityDecoder()
#         self.surrogate_loss_mlp = SurrogateLossMLP()
#         self.mse_loss = nn.MSELoss()

#     def forward(self, modalities):
#         encoder_input_modalities = torch.cat((modalities['sentinel2'], modalities['sentinel1']), dim=1)
#         input_embeddings = self.pretrained_encoder(encoder_input_modalities)
#         modality_reconstructions_raw = self.task_modality_decoder(input_embeddings)
#         modality_reconstructions = {'sentinel2': modality_reconstructions_raw[:, :12],
#                                     'sentinel1': modality_reconstructions_raw[:, 12:20],
#                                     'asterdem': modality_reconstructions_raw[:, 20:22],
#                                     'ethgch': modality_reconstructions_raw[:, 22:24]}
#         reconstruction_losses = {modality: self.mse_loss(reconstruction, globals()[modality]) for modality, reconstruction in modality_reconstructions.items()}
#         surrogate_loss = self.surrogate_loss_mlp(reconstruction_losses)

#         return downstream_task_prediction, modality_reconstructions, surrogate_loss

# class Model(LightningModule):
#     def __init__(self):
#         super().__init__()
#         self.model = TTAModel()
