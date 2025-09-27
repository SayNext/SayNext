"""
 * Copyright (c) 2025.
 * All rights reserved.
 * Code for SayNext project
"""
import warnings
from typing import Any, List, Optional, Tuple, Union
import os

import torch.distributed as dist
import torch.utils.checkpoint
import transformers
from saynext.conversation import get_conv_template
from saynext.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from saynext.model.phi3.modeling_phi3 import Phi3ForCausalLM
from peft import LoraConfig, get_peft_model
from torch import nn 
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from .configuration_internvl_chat import InternVLChatConfig
from .modeling_intern_vit import InternVisionModel

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))

# def compute_vae_loss(recon_x, x, mu, logvar):
#     """
#     Compute VAE Loss: Reconstruction Loss + KL Divergence Loss.
#     Args:
#         recon_x: Reconstructed input.
#         x: Original input.
#         mu: Mean of the latent space.
#         logvar: Log-variance of the latent space.
#     Returns:
#         Total VAE loss.
#     """
#     # Reconstruction loss (MSE)
#     recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    
#     # KL divergence loss
#     kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    
#     return recon_loss + kl_loss

def compute_vae_loss(recon_x, x, mu, logvar, beta=0.1):
    """
    Compute VAE Loss: Reconstruction Loss + KL Divergence Loss (with stabilization).
    Args:
        recon_x: Reconstructed input.
        x: Original input.
        mu: Mean of the latent space.
        logvar: Log-variance of the latent space.
        beta: Weight for KL divergence (default: 0.1).
    Returns:
        Stabilized total VAE loss.
    """
    # print(f"recon_x - min: {recon_x.min().item()}, max: {recon_x.max().item()}, contains NaN: {torch.isnan(recon_x).any().item()}")
    # print(f"x - min: {x.min().item()}, max: {x.max().item()} {x.mean()} {x.std()}, contains NaN: {torch.isnan(x).any().item()}")
    norm_x = (x - x.mean()) / (x.std() + 1e-6)
    # print(f"norm_x - min: {norm_x.min().item()}, max: {norm_x.max().item()}, contains NaN: {torch.isnan(norm_x).any().item()}")

    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(norm_x, recon_x, reduction='mean')
    # Clamp logvar to avoid extreme values
    # logvar = torch.clamp(logvar, min=-10, max=10)

    # KL divergence loss with beta scaling for stabilization
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss /= x.size(0)  # Normalize by batch size
    kl_loss /= mu.size(1) # NEW

    # kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=2)) # 和上面是等价的

    # Combine reconstruction and KL divergence losses
    vae_loss = recon_loss + beta * kl_loss
    print(f"recon_loss: {recon_loss.item()}, kl_loss: {kl_loss.item()}, vae_loss:{vae_loss.item()}")
    if torch.isnan(recon_loss).any() or torch.isnan(kl_loss).any():
        print("Warning: NaN detected in VAE loss components!")
    return vae_loss


# class VAE(nn.Module):
#     def __init__(self, latent_dim=64, hidden_channels=256):
#         """
#         输入: x.shape = [batch, 256, 4096]
#         """
#         super(VAE, self).__init__()
        
#         # Encoder 部分
#         # 先将输入转置为 [batch, 4096, 256]，将 4096 作为 in_channels，
#         # 用 1D 卷积在 token 维度（长度 256）上提取局部特征
#         self.enc_conv1 = nn.Conv1d(in_channels=4096, out_channels=hidden_channels, kernel_size=3, stride=2, padding=1)
#         self.enc_act1 = nn.LeakyReLU(0.2)
#         self.enc_conv2 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=2, padding=1)
#         self.enc_act2 = nn.LeakyReLU(0.2)

#         # 在第二个卷积层之后添加 BatchNorm1d 归一化层
#         self.enc_norm = nn.BatchNorm1d(hidden_channels)

#         # 经两层卷积后，token 维度从 256 依次缩小到 128 和 64
#         # 将 [batch, hidden_channels, 64] 展平后得到特征维度 hidden_channels*64
#         self.enc_fc = nn.Linear(hidden_channels * 64, latent_dim * 2)  # 输出 latent_dim*2 用于分别计算均值和 log方差
        
#         # Decoder 部分
#         self.dec_fc = nn.Linear(latent_dim, hidden_channels * 64)
#         # 重构后形状为 [batch, hidden_channels, 64]，然后经过反卷积逐步恢复 token 数量
#         self.dec_deconv1 = nn.ConvTranspose1d(in_channels=hidden_channels, out_channels=hidden_channels, 
#                                               kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.dec_act1 = nn.LeakyReLU(0.2)
#         self.dec_deconv2 = nn.ConvTranspose1d(in_channels=hidden_channels, out_channels=4096, 
#                                               kernel_size=3, stride=2, padding=1, output_padding=1)
#         # 最后再转置回来，使输出形状为 [batch, 256, 4096]
    
#     def reparameterize(self, mu, logvar, deterministic=False):
#         if deterministic:
#             return mu
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std
    
#     def encode(self, x):
#         # x shape: [batch, 256, 4096]
#         # 转置为 [batch, 4096, 256] 以适配 Conv1d
#         x = x.transpose(1, 2)
#         h = self.enc_act1(self.enc_conv1(x))   # shape: [batch, hidden_channels, 128]
#         h = self.enc_act2(self.enc_conv2(h))     # shape: [batch, hidden_channels, 64]
#         h = self.enc_norm(h)                     # 加入归一化层，稳定每个通道的分布
#         # 展平特征
#         h = h.view(h.size(0), -1)                # shape: [batch, hidden_channels * 64]
#         h = self.enc_fc(h)                       # shape: [batch, latent_dim*2]
#         mu, logvar = h.chunk(2, dim=1)           # 均值和 log方差各 shape: [batch, latent_dim]
#         return mu, logvar
    
#     def decode(self, z):
#         h = self.dec_fc(z)                       # shape: [batch, hidden_channels * 64]
#         # 恢复到卷积特征图形状
#         h = h.view(z.size(0), -1, 64)             # shape: [batch, hidden_channels, 64]
#         h = self.dec_act1(self.dec_deconv1(h))     # shape: [batch, hidden_channels, 128]
#         h = self.dec_deconv2(h)                    # shape: [batch, 4096, 256]
#         # 转置回原始形状 [batch, 256, 4096]
#         x_hat = h.transpose(1, 2)
#         return x_hat
    
#     def forward(self, x, deterministic=False):
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar, deterministic)
#         x_hat = self.decode(z)
#         return x_hat, mu, logvar

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim):
        super(VAE, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.norm_input = nn.LayerNorm(hidden_dim, eps=1e-2)
        self.FC_input2 = nn.Linear(hidden_dim, latent_dim)
        self.norm_input2 = nn.LayerNorm(latent_dim, eps=1e-2)
        self.FC_mean = nn.Linear(latent_dim, latent_dim)
        self.FC_var = nn.Linear(latent_dim, latent_dim)

        self.FC_hidden = nn.Linear(latent_dim, output_dim)
        self.norm_hidden = nn.LayerNorm(output_dim)
        # self.FC_hidden = nn.Linear(latent_dim, latent_dim)
        # self.norm_hidden = nn.LayerNorm(latent_dim)
        # self.FC_hidden2 = nn.Linear(latent_dim, output_dim)
        # self.norm_hidden2 = nn.LayerNorm(output_dim)
        self.FC_output = nn.Linear(output_dim, output_dim)

        # 使用 nn.GELU 激活函数替代原来的 LeakyReLU
        self.activation = nn.GELU()

    def encoder(self, x):
        # x: [batch, input_dim]
        h = self.FC_input(x)                  # [16, 256, hidden_dim]，例如 [16, 256, 3584]
        h = self.norm_input(h)                # [16, 256, hidden_dim]，例如 [16, 256, 3584]
        h = self.activation(h)                # [16, 256, hidden_dim]，例如 [16, 256, 3584]
        h = self.FC_input2(h)                 # [16, 256, latent_dim]，例如 [16, 256, 3072]
        h = self.norm_input2(h)               # [16, 256, latent_dim]，例如 [16, 256, 3072]
        h = self.activation(h)                # [16, 256, latent_dim]，例如 [16, 256, 3072]
        mean = self.FC_mean(h)                # [16, 256, latent_dim]，例如 [16, 256, 3072]
        log_var = self.FC_var(h)              # [16, 256, latent_dim]，例如 [16, 256, 3072]
        return mean, log_var

    def reparameterization(self, mean, log_var, deterministic=False):
        if deterministic:
            return mean  
        else:
            epsilon = torch.randn_like(log_var)   
            z = mean + torch.exp(0.5 * log_var) * epsilon  
            return z

    def decoder(self, z, return_h=False):
        # z: [batch, token_num, latent_dim]，例如 [16, 256, 3072]
        h = self.FC_hidden(z)                 # [16, 256, output_dim]，例如 [16, 256, 3072]
        h = self.norm_hidden(h)               # [16, 256, output_dim]，例如 [16, 256, 3072]
        h = self.activation(h)                # [16, 256, output_dim]，例如 [16, 256, 3072]
        # h = self.FC_hidden2(h)                # [16, 256, output_dim]，例如 [16, 256, 4096]
        # h = self.norm_hidden2(h)              # [16, 256, output_dim]，例如 [16, 256, 4096]
        # h = self.activation(h)                # [16, 256, output_dim]，例如 [16, 256, 4096]
        x_hat = self.FC_output(h)             # [16, 256, output_dim]，例如 [16, 256, 4096]
        return x_hat

    def forward(self, x, deterministic=False, return_h=False):
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, log_var, deterministic=deterministic)
        x_hat = self.decoder(z)
        return x_hat, mean, log_var, z

        


# class VAE(nn.Module):
#     def __init__(self, input_dim=4096, latent_dim=64, hidden_dim=1024):
#         """
#         Variational Autoencoder (VAE) for encoding and reconstructing visual features.
#         Args:
#             input_dim (int): Dimensionality of the input features (e.g., 4096 from vit_embeds).
#             latent_dim (int): Dimensionality of the latent space (default: 64).
#             hidden_dim (int): Dimensionality of the hidden layers (default: 1024).
#         """
#         super(VAE, self).__init__()

#         # Encoder network
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             # nn.ReLU(),
#             nn.GELU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.GELU(),
#             # nn.ReLU()
#         )

#         # Latent space: mean and log-variance
#         self.fc_mu = nn.Linear(hidden_dim, latent_dim)
#         self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

#         # Decoder network
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, input_dim)
#         )

#         # Initialize weights with Xavier initialization for stability
#         self.encoder.apply(self.weights_init)
#         self.fc_mu.apply(self.weights_init)
#         self.fc_logvar.apply(self.weights_init)
#         self.decoder.apply(self.weights_init)

#     def weights_init(self, m):
#         """Xavier initialization for Linear layers."""
#         if isinstance(m, nn.Linear):
#             nn.init.xavier_uniform_(m.weight)
#             nn.init.constant_(m.bias, 0)

#     def encode(self, x):
#         """Encodes input to latent space parameters (mean and log-variance) with stabilization."""
#         # Normalize input to avoid extreme values
#         print(f"before normalization x - min: {x.min().item()}, max: {x.max().item()}, contains NaN: {torch.isnan(x).any().item()}")
#         print(f"x shape: {x.shape}") 
#         x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)
#         print(f"Normalized x - min: {x.min().item()}, max: {x.max().item()}, contains NaN: {torch.isnan(x).any().item()}")

#         h = x
#         for i, layer in enumerate(self.encoder):
#             h = layer(h)
#             print(f"Layer {i} - min: {h.min().item()}, max: {h.max().item()}, contains NaN: {torch.isnan(h).any().item()}")

#         # Compute mean and log-variance, then clamp to stabilize
#         mu = torch.clamp(self.fc_mu(h), min=-5, max=5)
#         logvar = torch.clamp(self.fc_logvar(h), min=-5, max=5)
#         print(f"mu - min: {mu.min().item()}, max: {mu.max().item()}, contains NaN: {torch.isnan(mu).any().item()}")
#         print(f"logvar - min: {logvar.min().item()}, max: {logvar.max().item()}, contains NaN: {torch.isnan(logvar).any().item()}")

#         return mu, logvar

#     def reparameterize(self, mu, logvar):
#         """Reparameterization trick to sample from N(mu, var) with stabilization."""
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def decode(self, z):
#         """Decodes from latent space back to the original feature space."""
#         return self.decoder(z)

#     def forward(self, x):
#         """Full forward pass: encode, reparameterize, and decode."""
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         recon_x = self.decode(z)
#         return recon_x, mu, logvar



class VAE_InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'InternLM2DecoderLayer',
                         'Phi3DecoderLayer', 'Qwen2DecoderLayer']
    _supports_flash_attn_2 = True

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.llm_arch_name = config.llm_config.architectures[0]

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Phi3ForCausalLM':
                self.language_model = Phi3ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        # self.mlp1 = nn.Sequential(
        #     nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
        #     nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
        #     nn.GELU(),
        #     nn.Linear(llm_hidden_size, llm_hidden_size)
        # )

        # Instantiate VAE for visual feature encoding
        # self.vae = VAE(input_dim=4096, hidden_dim=1024, latent_dim=64, output_dim=4096)  # VAE input matches vit_embeds size (4096)
        # self.vae = VAE(latent_dim=64, hidden_channels=256)
        self.vae = VAE(input_dim=4096, hidden_dim=3584, latent_dim=3072, output_dim=4096)

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        if hasattr(config, 'system_message'):
            self.system_message = config.system_message
        else:
            self.system_message = self.conv_template.system_message
        self.num_samples = 0

        if config.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)

        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)



    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        # Determine the target modules based on the architecture of the language model
        if self.llm_arch_name == 'InternLM2ForCausalLM':
            target_modules = ['attention.wqkv', 'attention.wo', 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']
        elif self.llm_arch_name == 'Phi3ForCausalLM':
            target_modules = ['mlp.down_proj', 'mlp.gate_up_proj', 'self_attn.o_proj', 'self_attn.qkv_proj']
        elif self.llm_arch_name in ['Qwen2ForCausalLM', 'LlamaForCausalLM']:
            target_modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                              'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']
        else:
            raise NotImplemented
        lora_config = LoraConfig(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM'
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            ignore_flag = False
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
            ignore_flag = True

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if ignore_flag:
                loss = loss * 0.0

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        # vit_embeds = self.mlp1(vit_embeds)
        x_hat, mean, log_var, z = self.vae(vit_embeds)
        # loss =
        vit_embeds = x_hat
        return vit_embeds #, loss

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep)[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()
        history.append((question, response))

        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        gen_config = GenerationConfig(
            max_length=200,           # 最大生成长度
            min_length=10,            # 最小生成长度
            do_sample=True,           # 是否采用采样策略
            temperature=0.7,          # 采样温度
            top_k=50,                 # Top-k 采样
            top_p=0.95,               # Top-p（nucleus）采样
            no_repeat_ngram_size=3,   # 禁止重复 n-gram（例如连续3个词）
            num_beams=5,              # beam search 的 beam 数量
            early_stopping=True,      # beam search 提前停止条件
            repetition_penalty=1.2,   # 重复惩罚
    )

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=gen_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs
