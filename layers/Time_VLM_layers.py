import einops
import torch
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
from sqlite3 import Time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def time_series_to_simple_image(x_enc, image_size, context_len, periodicity):
    """
    Convert time series data into 3-channel image tensors.
    
    Args:
        x_enc (torch.Tensor): Input time series data of shape [B, seq_len, nvars].
        image_size (int): Size of the output image (height and width).
        context_len (int): Length of the time series sequence.
        periodicity (int): Periodicity used to reshape the time series into 2D.
        
    Returns:
        torch.Tensor: Image tensors of shape [B, 3, H, W].
    """
    B, seq_len, nvars = x_enc.shape  # 获取输入形状

    # Adjust padding to make context_len a multiple of periodicity
    pad_left = 0
    if context_len % periodicity != 0:
        pad_left = periodicity - context_len % periodicity

    # Rearrange to [B, nvars, seq_len]
    x_enc = einops.rearrange(x_enc, 'b s n -> b n s')

    # Pad the time series
    x_pad = F.pad(x_enc, (pad_left, 0), mode='replicate')
    
    # Reshape to [B * nvars, 1, f, p]
    x_2d = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=periodicity)
    
    # Resize the time series data
    x_resized_2d = F.interpolate(x_2d, size=(image_size, image_size), mode='bilinear', align_corners=False)

    # Convert to 3-channel image
    images = einops.repeat(x_resized_2d, 'b 1 h w -> b c h w', c=3)  # [B * nvars, 3, H, W]

    # Reshape back to [B, nvars, 3, H, W] and average over nvars
    images = einops.rearrange(images, '(b n) c h w -> b n c h w', b=B, n=nvars)  # [B, nvars, 3, H, W]
    images = images.mean(dim=1)  # Average over nvars to get [B, 3, H, W]
    
    return images


def time_series_to_image_with_fft_and_wavelet(x_enc, image_size, context_len, periodicity):
    """
    Convert time series data into 3-channel image tensors using FFT and Wavelet transforms.
    
    Args:
        x_enc (torch.Tensor): Input time series data of shape [B, seq_len, nvars].
        image_size (int): Size of the output image (height and width).
        context_len (int): Length of the time series sequence.
        periodicity (int): Periodicity used to reshape the time series into 2D.
        
    Returns:
        torch.Tensor: Image tensors of shape [B, 3, H, W].
    """
    def _apply_fourier_transform(x_2d):
        """
        Apply Fourier transform to the input 2D time series data.
        """
        x_fft = torch.fft.fft(x_2d, dim=-1)
        x_fft_abs = torch.abs(x_fft)  # Take the magnitude part of the Fourier transform
        return x_fft_abs

    def _apply_wavelet_transform(x_2d):
        """
        Apply wavelet transform to the input 2D time series data.
        """
        dwt = DWTForward(J=1, wave='haar')
        # cA: Low-frequency components, cD: High-frequency components
        cA, cD = dwt(x_2d)  # [B * nvars, 1, f, p]
        cD_reshaped = cD[0].squeeze(1)  # [B * nvars, 3, f, p]
        # Concatenate low-frequency and high-frequency components
        wavelet_result = torch.cat([cA, cD_reshaped], dim=1)  # [B * nvars, 4, f, p]
        # Average across the channel dimension to reduce to 1 channel
        wavelet_result = wavelet_result.mean(dim=1, keepdim=True)  # [B * nvars, 1, f, p]
        return wavelet_result
    
    B, seq_len, nvars = x_enc.shape  # 获取输入形状

    # Adjust padding to make context_len a multiple of periodicity
    pad_left = 0
    if context_len % periodicity != 0:
        pad_left = periodicity - context_len % periodicity

    # Rearrange to [B, nvars, seq_len]
    x_enc = einops.rearrange(x_enc, 'b s n -> b n s')

    # Pad the time series
    x_pad = F.pad(x_enc, (pad_left, 0), mode='replicate')
    
    # Reshape to [B * nvars, 1, f, p]
    x_2d = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=periodicity)
    
    # Resize the time series data
    x_resized_2d = F.interpolate(x_2d, size=(image_size, image_size), mode='bilinear', align_corners=False)

    # Apply Fourier transform or wavelet transform
    x_fft = _apply_fourier_transform(x_2d)
    x_wavelet = _apply_wavelet_transform(x_2d)
    # Resize the Fourier or wavelet transformed data as image input using interpolation
    x_resized_fft = F.interpolate(x_fft, size=(image_size, image_size), mode='bilinear', align_corners=False)
    x_resized_wavelet = F.interpolate(x_wavelet, size=(image_size, image_size), mode='bilinear', align_corners=False)
    # Concatenate along the channel dimension to form a 3-channel image
    images = torch.concat([x_resized_2d, x_resized_fft, x_resized_wavelet], dim=1)  # [B * nvars, 3, H, W]

    # Reshape back to [B, nvars, 3, H, W] and average over nvars
    images = einops.rearrange(images, '(b n) c h w -> b n c h w', b=B, n=nvars)  # [B, nvars, 3, H, W]
    images = images.mean(dim=1)  # Average over nvars to get [B, 3, H, W]
    
    return images

def generate_multi_channel_time_series_prompt(self, x_enc, description, pred_len, seq_len, top_k=5):
    """
    Generate text prompts for the language model based on time series data.
    Each variable in the time series will have its own prompt.

    Args:
        x_enc (torch.Tensor): Input time series data, shape of [batch_size, seq_len, n_vars].
        description (str): Description of the dataset.
        pred_len (int): Prediction length.
        seq_len (int): Sequence length.
        top_k (int): Number of top variables to highlight (optional).

    Returns:
        List[str]: A list of prompts for each variable in each batch, with length B * n_vars.
    """
    B, T, n_vars = x_enc.shape  # Get batch size, sequence length, and number of variables

    # Initialize a list to store prompts for each variable in each batch
    prompts = []
    
    # Calculate statistics for each batch and each variable
    for b in range(B):
        for var in range(n_vars):
            # Extract the time series for the current variable
            var_series = x_enc[b, :, var]  # [seq_len]

            # Calculate statistics for the current variable
            min_value = torch.min(var_series).item()  # Minimum value
            max_value = torch.max(var_series).item()  # Maximum value
            median_value = torch.median(var_series).item()  # Median value
            trend = var_series.diff(dim=0).sum().item()  # Trend

            # Determine the trend direction
            trend_direction = "upward" if trend > 0 else "downward"

            # Generate prompt for the current variable
            prompt_parts = [
                f"The time series is converted into an image using 1D and 2D convolutional layers, highlighting trends, periodic patterns, and multi-scale features for forecasting.",
                f"Dataset: {description}",
                f"Task: Forecast the next {pred_len} steps using the past {seq_len} steps.",
                f"Input statistics: min value = {min_value:.3f}, max value = {max_value:.3f}, median value = {median_value:.3f}, the overall trend is {trend_direction}."
            ]
            prompt = " ".join(prompt_parts)
            prompt = prompt[:self.vlm_manager.max_input_text_length] if len(prompt) > self.vlm_manager.max_input_text_length else prompt
            prompts.append(prompt)

    return prompts

def generate_simple_multi_channel_time_series_prompt(self, x_enc, description, pred_len, seq_len, top_k=5):
    """
    Generate text prompts for the language model based on time series data.
    Each variable in the time series will have the same fixed prompt.

    Args:
        x_enc (torch.Tensor): Input time series data, shape of [batch_size, seq_len, n_vars].
        description (str): Description of the dataset.
        pred_len (int): Prediction length.
        seq_len (int): Sequence length.
        top_k (int): Number of top variables to highlight (optional).

    Returns:
        List[str]: A list of fixed prompts for each variable in each batch, with length B * n_vars.
    """
    B, T, n_vars = x_enc.shape  # Get batch size, sequence length, and number of variables

    # Fixed prompt content
    fixed_prompt = "The time series is converted into an image using 1D and 2D convolutional layers, highlighting trends, periodic patterns, and multi-scale features for forecasting."

    # Ensure the prompt does not exceed the maximum input text length
    fixed_prompt = fixed_prompt[:self.vlm_manager.max_input_text_length] if len(fixed_prompt) > self.vlm_manager.max_input_text_length else fixed_prompt

    # Repeat the fixed prompt for each variable in each batch
    prompts = [fixed_prompt] * (B * n_vars)

    return prompts

class TimeSeriesVisualizer:
    """Visualization tools for time series to image conversion"""
    
    @staticmethod
    def plot_feature_maps(features, title="Feature Maps"):
        """Plot feature maps from intermediate layers"""
        if not isinstance(features, torch.Tensor):
            return
            
        # Convert to numpy and normalize
        features = features.detach().cpu().numpy()
        features = (features - features.min()) / (features.max() - features.min() + 1e-8)
        
        # Plot first batch item
        num_channels = min(4, features.shape[1])
        if num_channels == 1:
            plt.figure(figsize=(5, 5))
            plt.imshow(features[0, 0], cmap='viridis')
            plt.axis('on')
            plt.title('Channel 1')
        else:
            fig, axes = plt.subplots(1, num_channels, figsize=(15, 5))
            for i, ax in enumerate(axes):
                ax.imshow(features[0, i], cmap='viridis')
                ax.axis('off')
                ax.set_title(f'Channel {i+1}')
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save figure
        import os
        os.makedirs('ts-images/ts-visualizer', exist_ok=True)
        filename = f"ts-images/ts-visualizer/{title.replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {filename}")
        plt.close()

    @staticmethod
    def plot_attention(attention_map, title="Attention Map"):
        """Plot attention weights"""
        if not isinstance(attention_map, torch.Tensor):
            return
            
        attention_map = attention_map.detach().cpu().numpy()
        plt.figure(figsize=(8, 8))
        plt.imshow(attention_map[0, 0], cmap='viridis')
        plt.colorbar()
        plt.title(title)
        plt.show()

class LearnableTimeSeriesToImage(nn.Module):
    """Learnable module to convert time series data into image tensors"""
    
    def __init__(self, input_dim, hidden_dim, output_channels, image_size, periodicity):
        super(LearnableTimeSeriesToImage, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels
        self.image_size = image_size
        self.periodicity = periodicity

        # 1D convolutional layer
        self.conv1d = nn.Conv1d(in_channels=4, out_channels=hidden_dim, kernel_size=3, padding=1)

        # 2D convolution layers
        self.conv2d_1 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim // 2, kernel_size=3, padding=1)
        self.conv2d_2 = nn.Conv2d(in_channels=hidden_dim // 2, out_channels=output_channels, kernel_size=3, padding=1)


    def forward(self, x_enc):
        """Convert input time series to image tensor [B, output_channels, H, W]"""
        B, L, D = x_enc.shape
        
        # Generate periodicity encoding (sin/cos)
        time_steps = torch.arange(L, dtype=torch.float32).unsqueeze(0).repeat(B, 1).to(x_enc.device)
        periodicity_encoding = torch.cat([
            torch.sin(time_steps / self.periodicity * (2 * torch.pi)).unsqueeze(-1),
            torch.cos(time_steps / self.periodicity * (2 * torch.pi)).unsqueeze(-1)
        ], dim=-1)
        periodicity_encoding = periodicity_encoding.unsqueeze(-2).repeat(1, 1, D, 1)  # [B, L, D, 2]
        
        # FFT frequency encoding (magnitude)
        x_fft = torch.fft.rfft(x_enc, dim=1)
        x_fft_mag = torch.abs(x_fft)
        if x_fft_mag.shape[1] < L:
            pad = torch.zeros(B, L - x_fft_mag.shape[1], D, device=x_enc.device, dtype=x_fft_mag.dtype)
            x_fft_mag = torch.cat([x_fft_mag, pad], dim=1)
        x_fft_mag = x_fft_mag.unsqueeze(-1)  # [B, L, D, 1]

        # Combine all features: raw + FFT + periodic
        x_enc = x_enc.unsqueeze(-1)  # [B, L, D, 1]
        x_enc = torch.cat([x_enc, x_fft_mag, periodicity_encoding], dim=-1)  # [B, L, D, 4]

        # Reshape for 1D convolution
        x_enc = x_enc.permute(0, 2, 3, 1)  # [B, D, 4, L]
        x_enc = x_enc.reshape(B * D, 4, L)  # [B*D, 4, L]
        x_enc = self.conv1d(x_enc)  # [B*D, hidden_dim, L]
        x_enc = x_enc.reshape(B, D, self.hidden_dim, L)  # [B, D, hidden_dim, L]

        # 2D Convolution processing
        x_enc = x_enc.permute(0, 2, 1, 3)  # [B, hidden_dim, D, L]
        x_enc = F.relu(self.conv2d_1(x_enc))
        x_enc = F.relu(self.conv2d_2(x_enc))
        
        # Resize to target image size
        x_enc = F.interpolate(x_enc, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        
        return x_enc  # [B, output_channels, H, W]


class MultiChannalLearnableTimeSeriesToImage(nn.Module):
    """Learnable module to convert time series data into image tensors"""
    
    def __init__(self, input_dim, hidden_dim, output_channels, image_size, periodicity):
        super(MultiChannalLearnableTimeSeriesToImage, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels
        self.image_size = image_size
        self.periodicity = periodicity

        # 1D convolutional layer
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)

        # 2D convolution layers
        self.conv2d_1 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim // 2, kernel_size=3, padding=1)
        self.conv2d_2 = nn.Conv2d(in_channels=hidden_dim // 2, out_channels=output_channels, kernel_size=3, padding=1)


    def forward(self, x_enc):
        """Convert input time series to image tensor [B, output_channels, H, W]"""
        B, L, D = x_enc.shape
        
        # Generate periodicity encoding (sin/cos)
        time_steps = torch.arange(L, dtype=torch.float32).unsqueeze(0).repeat(B, 1).to(x_enc.device)
        periodicity_encoding = torch.cat([
            torch.sin(time_steps / self.periodicity * (2 * torch.pi)).unsqueeze(-1),
            torch.cos(time_steps / self.periodicity * (2 * torch.pi)).unsqueeze(-1)
        ], dim=-1)
        periodicity_encoding = periodicity_encoding.unsqueeze(-2).repeat(1, 1, D, 1)  # [B, L, D, 2]
        
        # Combine raw data with periodic encoding
        x_enc = x_enc.unsqueeze(-1)  # [B, L, D, 1]
        x_enc = torch.cat([x_enc, periodicity_encoding], dim=-1)  # [B, L, D, 3]

        # Reshape for 1D convolution
        x_enc = x_enc.permute(0, 2, 3, 1)  # [B, D, 3, L]
        x_enc = x_enc.reshape(B * D, 3, L)  # [B*D, 3, L]
        x_enc = self.conv1d(x_enc)  # [B*D, hidden_dim, L]
        
        # Add channel dimension for 2D processing
        x_enc = x_enc.unsqueeze(2)  # [B*D, hidden_dim, 1, L]
        
        # 2D Convolution processing
        x_enc = F.relu(self.conv2d_1(x_enc))
        x_enc = F.relu(self.conv2d_2(x_enc))
        
        # Resize to target image size
        x_enc = F.interpolate(x_enc, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        
        return x_enc  # [B*D, output_channels, H, W]

class MultiscaleLearnableTimeSeriesToImage(nn.Module):
    """Enhanced learnable module for time series to image conversion with multi-scale features"""
    
    def __init__(self, input_dim, hidden_dim, output_channels, image_size, periodicity):
        super(MultiscaleLearnableTimeSeriesToImage, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels
        self.image_size = image_size
        self.periodicity = periodicity

        # Multi-scale 1D convolutions with different dilations
        self.conv1d_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1, dilation=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
                nn.ReLU()
            ),
        ])
        
        # Frequency domain convolution
        self.fft_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        
        # Residual connection
        self.residual = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        
        # 2D convolution blocks
        self.conv2d_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(hidden_dim // 2, output_channels, kernel_size=3, padding=1),
                nn.ReLU()
            ),
        ])
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim//8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim//8, hidden_dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv2d_2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        
        # Discrete wavelet transform
        self.dwt = DWTForward(J=1, wave='haar')
        
        # Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Conv1d(18*hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )

        # Projection layers
        self.proj_for_attention = nn.Conv2d(3, hidden_dim, kernel_size=1)
        self.final_proj_to_output = nn.Conv2d(hidden_dim, output_channels, kernel_size=1)

    def forward(self, x_enc):
        """Convert input time series to image tensor [B, output_channels, H, W]"""
        B, L, D = x_enc.shape
        
        # Generate periodicity encoding (sin/cos)
        time_steps = torch.arange(L, dtype=torch.float32).unsqueeze(0).repeat(B, 1).to(x_enc.device)
        periodicity_encoding = torch.cat([
            torch.sin(time_steps / self.periodicity * (2 * torch.pi)).unsqueeze(-1),
            torch.cos(time_steps / self.periodicity * (2 * torch.pi)).unsqueeze(-1)
        ], dim=-1)
        periodicity_encoding = periodicity_encoding.unsqueeze(-2).repeat(1, 1, D, 1)  # [B, L, D, 2]

        # Combine raw data with periodic encoding
        x_enc = x_enc.unsqueeze(-1)  # [B, L, D, 1]
        x_enc = torch.cat([x_enc, periodicity_encoding], dim=-1)  # [B, L, D, 3]

        # Reshape for multi-scale processing
        x_enc = x_enc.view(B * D, 3, L)

        # Multi-scale 1D convolution features
        conv_features = []
        for conv_block in self.conv1d_blocks:
            conv_features.append(conv_block(x_enc))
        x_enc = torch.cat(conv_features, dim=1)  # Concatenate along channel dim
        
        # Frequency domain processing
        freq_features = self.freq_processing(x_enc)
        
        # Downsample and fuse features
        x_enc_downsampled = x_enc[:, :, ::2]  # Match freq feature length
        x_enc = torch.cat([x_enc_downsampled, freq_features], dim=1)
        x_enc = self.feature_fusion(x_enc)

        # Prepare for 2D processing
        x_enc = x_enc.unsqueeze(2)  # Add spatial dimension

        # 2D convolution processing
        for conv2d_block in self.conv2d_blocks:
            x_enc = conv2d_block(x_enc)
        
        # Attention mechanism
        x_enc = self.proj_for_attention(x_enc)
        attention_map = self.attention(x_enc)
        x_enc = x_enc * attention_map  # Apply attention weights

        # Final projection and resizing
        x_enc = self.final_proj(x_enc)
        x_enc = self.final_proj_to_output(x_enc)
        x_enc = F.interpolate(x_enc, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        
        # Reshape and average over variables
        x_enc = x_enc.view(B, D, self.output_channels, self.image_size, self.image_size)
        x_enc = x_enc.mean(dim=1)  # [B, output_channels, H, W]
                
        return x_enc

    def freq_processing(self, x):
        """Process frequency domain features using FFT and wavelet transforms"""
        x = x.float()
        
        # FFT magnitude features
        x_fft = torch.fft.rfft(x, dim=-1)
        x_fft = torch.abs(x_fft)
        x_fft = x_fft[..., :x.shape[2] // 2]  # Match wavelet feature length
        
        # Wavelet transform features
        x_2d = x.unsqueeze(2)
        cA, cD = self.dwt(x_2d)
        cD_reshaped = cD[0].squeeze(3)  # [B*D, C, 3, L//2]
        wavelet_features = torch.cat([cA, cD_reshaped], dim=2)  # [B*D, C, 4, L//2]
        
        # Match wavelet dimensions
        x_fft = x_fft.unsqueeze(2)  # [B*D, C, 1, L//2]
        x_fft = x_fft.expand(-1, -1, 4, -1)  # [B*D, C, 4, L//2]
        
        # Combine frequency features
        freq_features = torch.cat([x_fft, wavelet_features], dim=1)
        freq_features = freq_features.view(freq_features.shape[0], -1, freq_features.shape[-1])
        
        return freq_features


    def final_proj(self, x):
        """Final projection with residual connection"""
        identity = x
        x = self.conv2d_2(x)
        return x + identity  # Residual connection
