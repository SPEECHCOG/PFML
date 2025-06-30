# -*- coding: utf-8 -*-
"""
Author: Einari Vaaras, einari.vaaras@tuni.fi, Tampere University
Speech and Cognition Research Group, https://webpages.tuni.fi/specog/index.html

The different models used for PFML pre-training and fine-tuning. The file contains
data modality-specific encoders for framed speech, multi-sensor IMU, and EEG data.

NOTE: For detailed descriptions regarding the input variables for the models, see
the configuration files.

"""

import numpy as np
import math
import time
import sys
import torch
from torch.nn import Module, Linear, Conv1d, BatchNorm1d, Dropout, GELU, LayerNorm, Parameter, Identity
from torch.nn import AvgPool1d, Conv2d, AvgPool2d, LeakyReLU, Tanh, ELU, Conv3d, AvgPool3d, BatchNorm2d
from torch.nn import ReLU, MaxPool1d
from transformer_encoder_pytorch import Transformer_encoder_base




class raw_audio_encoder_CNN(Module):
    """
    A four-layer CNN encoder (containing strided convolutions) for framed raw audio data.
    
    """
    
    def __init__(self,
                 conv_1_in_dim = 1,
                 conv_1_out_dim = 512,
                 conv_1_kernel_size = 10,
                 conv_1_stride = 5,
                 conv_1_padding = 3,
                 num_norm_features_1 = 512,
                 conv_2_in_dim = 512,
                 conv_2_out_dim = 512,
                 conv_2_kernel_size = 8,
                 conv_2_stride = 4,
                 conv_2_padding = 2,
                 num_norm_features_2 = 512,
                 conv_3_in_dim = 512,
                 conv_3_out_dim = 512,
                 conv_3_kernel_size = 4,
                 conv_3_stride = 2,
                 conv_3_padding = 1,
                 num_norm_features_3 = 512,
                 conv_4_in_dim = 512,
                 conv_4_out_dim = 512,
                 conv_4_kernel_size = 4,
                 conv_4_stride = 2,
                 conv_4_padding = 1,
                 num_norm_features_4 = 512,
                 pooling_kernel_size = 6,
                 pooling_zero_padding = 0,
                 normalization_type = 'layernorm',
                 pooling_type = 'average',
                 non_linearity_function = 'gelu',
                 dropout = 0.0):

        super().__init__()
        
        # Batch normalization normalizes each feature separately across all batch samples
        if normalization_type == 'batchnorm':
            normalization_layer = BatchNorm1d
        
        # Layer normalization normalizes each each batch sample separately across all features
        elif normalization_type == 'layernorm':
            normalization_layer = LayerNorm
            
        elif normalization_type == None:
            normalization_layer = Identity
        else:
            sys.exit(f'Wrong value for argument "normalization_type": {normalization_type}')
        
        self.conv_layer_1 = Conv1d(in_channels=conv_1_in_dim, out_channels=conv_1_out_dim, kernel_size=conv_1_kernel_size,
                                   stride=conv_1_stride, padding=conv_1_padding)
        self.normalization_1 = normalization_layer(num_norm_features_1)
        
        self.conv_layer_2 = Conv1d(in_channels=conv_2_in_dim, out_channels=conv_2_out_dim, kernel_size=conv_2_kernel_size,
                                   stride=conv_2_stride, padding=conv_2_padding)
        self.normalization_2 = normalization_layer(num_norm_features_2)
        
        self.conv_layer_3 = Conv1d(in_channels=conv_3_in_dim, out_channels=conv_3_out_dim, kernel_size=conv_3_kernel_size,
                                   stride=conv_3_stride, padding=conv_3_padding)
        self.normalization_3 = normalization_layer(num_norm_features_3)
        
        self.conv_layer_4 = Conv1d(in_channels=conv_4_in_dim, out_channels=conv_4_out_dim, kernel_size=conv_4_kernel_size,
                                   stride=conv_4_stride, padding=conv_4_padding)
        self.normalization_4 = normalization_layer(num_norm_features_4)
        
        if pooling_type == 'average':
            self.pooling = AvgPool1d(kernel_size=pooling_kernel_size, padding=pooling_zero_padding)
        elif pooling_type == 'max':
            self.pooling = MaxPool1d(kernel_size=pooling_kernel_size, padding=pooling_zero_padding)
        elif pooling_type == None:
            self.pooling = Identity()
        else:
            sys.exit(f'Wrong value for argument "pooling_type": {pooling_type}')
        
        if non_linearity_function == 'relu':
            self.non_linearity = ReLU()
        elif non_linearity_function == 'elu':
            self.non_linearity = ELU()
        elif non_linearity_function == 'gelu':
            self.non_linearity = GELU()
        else:
            sys.exit(f'Wrong value for argument "non_linearity_function": {non_linearity_function}')
        
        self.dropout = Dropout(dropout)
        self.normalization_type = normalization_type
    
    
    def forward(self, X_batch):
        
        # X_batch is now of size [batch_size, num_frames, frame_len]
        X_output = []
        
        # We go through each frame of a sequence and produce an encoding
        for i in range(X_batch.size()[0]):
            X = X_batch[i, :, :]
            
            # Make the input X of size [num_frames, frame_len] into size [num_frames, 1, frame_len]
            # by adding a dummy dimension (number of channels)
            # --> with default values e.g. from torch.Size([298, 480]) into torch.Size([298, 1, 480])
            X = X.unsqueeze(1)
            
            if self.normalization_type == 'layernorm':
                X = self.dropout(self.non_linearity(self.normalization_1(self.conv_layer_1(X).permute(0, 2, 1)).permute(0, 2, 1)))
                X = self.dropout(self.non_linearity(self.normalization_2(self.conv_layer_2(X).permute(0, 2, 1)).permute(0, 2, 1)))
                X = self.dropout(self.non_linearity(self.normalization_3(self.conv_layer_3(X).permute(0, 2, 1)).permute(0, 2, 1)))
                X = self.dropout(self.pooling(self.non_linearity(self.normalization_4(self.conv_layer_4(X).permute(0, 2, 1)).permute(0, 2, 1))))
            else:
                X = self.dropout(self.non_linearity(self.normalization_1(self.conv_layer_1(X))))
                X = self.dropout(self.non_linearity(self.normalization_2(self.conv_layer_2(X))))
                X = self.dropout(self.non_linearity(self.normalization_3(self.conv_layer_3(X))))
                X = self.dropout(self.pooling(self.non_linearity(self.normalization_4(self.conv_layer_4(X)))))
            
            X = X.squeeze()
            
            # X is now of size [num_frames, conv_4_out_dim]
            # --> with default values torch.Size([298, 512])
            X_output.append(X)
        
        # X_output is now of size [batch_size, num_frames, conv_4_out_dim]
        X_output = torch.stack(X_output, dim=0)
        
        return X_output






class SENSOR_MODULE_v3(Module):
    """
    A four-layer CNN sensor encoder for multi-sensor IMU data that combines the raw accelerometer
    and gyroscope signals at a frame-level and outputs latent representations for each input frame.
    This same CNN encoder was used by Airaksinen et al. (2022) in
      https://www.nature.com/articles/s43856-022-00131-6
    and the present implementation is based on Airaksinen's TensorFlow implementation.
    
    """
    def __init__(self,
                 s_channels = 24,
                 input_channels = 120,
                 latent_channels = 70,
                 output_channels = 140,
                 dropout = 0.3,
                 conv_1_kernel_size = (1,3,11),
                 conv_2_kernel_size = (1,3,5),
                 conv_3_kernel_size = (1,4),
                 conv_4_kernel_size = (1,4),
                 conv_1_stride = (1,3,6),
                 conv_2_stride = (1,1,2),
                 conv_3_stride = 1,
                 conv_4_stride = 1,
                 conv_1_zero_padding = (0,0,0),
                 conv_2_zero_padding = (0,1,2),
                 conv_3_zero_padding = 'same',
                 conv_4_zero_padding = 'valid',
                 pooling_1_zero_padding = (0,0,0),
                 pooling_1_kernel_size = (1,1,10),
                 pooling_2_zero_padding = (0,0),
                 pooling_2_kernel_size = (1,4),
                 normalization_type = 'layernorm'):

        super().__init__()
        
        # Batch normalization normalizes each feature separately across all batch samples
        if normalization_type == 'batchnorm':
            normalization_layer = BatchNorm1d
        
        # Layer normalization normalizes each each batch sample separately across all features
        elif normalization_type == 'layernorm':
            normalization_layer = LayerNorm
            
        elif normalization_type == None:
            normalization_layer = Identity
        else:
            sys.exit(f'Wrong value for argument "normalization_type": {normalization_type}')
        
        self.r = np.int32(s_channels / 2) # Default: 12
        
        self.conv_1_1 = Conv3d(in_channels=1, out_channels=latent_channels, 
                             kernel_size=conv_1_kernel_size, stride=conv_1_stride,
                             padding=conv_1_zero_padding, bias=True)
        self.conv_1_2 = Conv3d(in_channels=1, out_channels=latent_channels, 
                             kernel_size=conv_1_kernel_size, stride=conv_1_stride,
                             padding=conv_1_zero_padding, bias=True)
        self.conv_1_3 = Conv3d(in_channels=1, out_channels=latent_channels, 
                             kernel_size=conv_1_kernel_size, stride=conv_1_stride,
                             padding=conv_1_zero_padding, bias=True)
        
        self.conv_2_1 = Conv3d(in_channels=latent_channels, out_channels=latent_channels, 
                             kernel_size=conv_2_kernel_size, stride=conv_2_stride,
                             padding=conv_2_zero_padding, bias=True)
        self.conv_2_2 = Conv3d(in_channels=latent_channels, out_channels=latent_channels, 
                             kernel_size=conv_2_kernel_size, stride=conv_2_stride,
                             padding=conv_2_zero_padding, bias=True)
        self.conv_2_3 = Conv3d(in_channels=latent_channels, out_channels=latent_channels, 
                             kernel_size=conv_2_kernel_size, stride=conv_2_stride,
                             padding=conv_2_zero_padding, bias=True)
        
        self.conv_3 = Conv2d(in_channels=latent_channels, out_channels=output_channels, 
                             kernel_size=conv_3_kernel_size, stride=conv_3_stride,
                             padding=conv_3_zero_padding, bias=True)
        
        self.conv_4 = Conv2d(in_channels=output_channels, out_channels=output_channels, 
                             kernel_size=conv_4_kernel_size, stride=conv_4_stride,
                             padding=conv_4_zero_padding, bias=True)
        
        self.pooling_1 = AvgPool3d(kernel_size=pooling_1_kernel_size, padding=pooling_1_zero_padding)
        self.pooling_2 = AvgPool2d(kernel_size=pooling_2_kernel_size, padding=pooling_2_zero_padding)
        
        self.normalization = normalization_layer(output_channels)
        self.normalization_type = normalization_type
        
        self.tanh = Tanh()
        self.lrelu = LeakyReLU()
        
        self.dropout = Dropout(dropout)


    def _conv_module_1(self, X):
        
        # X is now of shape [batch_size, 1, Nframes, 3*x, 120]
        X = self.tanh(self.conv_1_1(X)) # X is now of shape [batch_size, latent_channels, Nframes, x, 19]
        X = self.lrelu(self.conv_2_1(X))
        X = torch.squeeze(self.pooling_1(X), dim=4) # X is now of shape [batch_size, latent_channels, Nframes, x]
        
        return X
    
    def _conv_module_2(self, X):
        
        # X is now of shape [Nframes, 1, 3*x, 120]
        X = self.tanh(self.conv_1_2(X)) # X is now of shape [batch_size, latent_channels, Nframes, x, 19]
        X = self.lrelu(self.conv_2_2(X))
        X = torch.squeeze(self.pooling_1(X), dim=4) # X is now of shape [batch_size, latent_channels, Nframes, x]
        
        return X
    
    def _conv_module_3(self, X):
        
        # X is now of shape [Nframes, 1, 6*x, wl]
        X = self.tanh(self.conv_1_3(X)) # X is now of shape [batch_size, latent_channels, Nframes, x, 19]
        X = self.lrelu(self.conv_2_3(X))
        X = torch.squeeze(self.pooling_1(X), dim=4) # X is now of shape [batch_size, latent_channels, Nframes, 2*x]
        
        return X
    
    def forward(self, X):
        
        X = self.dropout(X) # X is of size [batch_size, Nframes, s_channels, input_channels]
        X = torch.unsqueeze(X, dim=1) # X is of size [batch_size, 1, Nframes, s_channels, input_channels]
        acc_data = X[:, :, :, :(self.r), :] # acc_data is of size [batch_size, 1, Nframes, r, input_channels]
        gyro_data = X[:, :, :, (self.r):, :] # gyro_data is of size [batch_size, 1, Nframes, r, input_channels]
        
        # Get convolution embeddings for acceleration and gyro, both separately and together
        acc_emb = self._conv_module_1(acc_data) # acc_emb is of size [batch_size, latent_channels, Nframes, 4]
        gyro_emb = self._conv_module_2(gyro_data) # gyro_emb is of size [batch_size, latent_channels, Nframes, 4]
        both_emb = self._conv_module_3(X) # X is of size [batch_size, latent_channels, Nframes, 8]
        
        # We fuse acceleration and gyro
        X_fused = torch.cat((acc_emb, gyro_emb, both_emb), dim=3) # X_fused is of size [batch_size, latent_channels, Nframes, 16]
        if self.normalization_type == 'layernorm':
            X_fused = self.pooling_2(self.lrelu(self.normalization(self.conv_3(X_fused).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))
        else:
            X_fused = self.pooling_2(self.lrelu(self.normalization(self.conv_3(X_fused))))
        # Now X_fused is of size [batch_size, Nframes, output_channels, 4]
        
        # We fuse sensors
        if self.normalization_type == 'layernorm':
            X_output = torch.squeeze(self.lrelu(self.normalization(self.conv_4(X_fused).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)), dim=3)
        else:
            X_output = torch.squeeze(self.lrelu(self.normalization(self.conv_4(X_fused))), dim=3)
        X_output = X_output.permute(0, 2, 1) # X_output is of size [batch_size, Nframes, output_channels]   
        
        return X_output







class eeg_encoder_CNN(Module):
    """
    A three-layer CNN encoder (containing strided convolutions) for framed EEG data.
    
    """
    
    def __init__(self,
                 conv_1_in_dim = 1,
                 conv_1_out_dim = 128,
                 conv_1_kernel_size = (1,10),
                 conv_1_stride = (1,5),
                 conv_1_padding = (0,3),
                 num_norm_features_1 = 128,
                 conv_2_in_dim = 128,
                 conv_2_out_dim = 128,
                 conv_2_kernel_size = (1,8),
                 conv_2_stride = (1,5),
                 conv_2_padding = (0,2),
                 num_norm_features_2 = 128,
                 conv_3_in_dim = 128,
                 conv_3_out_dim = 128,
                 conv_3_kernel_size = (1,4),
                 conv_3_stride = (1,3),
                 conv_3_padding = (0,1),
                 num_norm_features_3 = 128,
                 pooling_kernel_size = (1,5),
                 pooling_zero_padding = (0,0),
                 normalization_type = 'layernorm',
                 non_linearity_function = 'gelu',
                 dropout = 0.0):

        super().__init__()
        
        # Batch normalization normalizes each feature separately across all batch samples
        if normalization_type == 'batchnorm':
            normalization_layer = BatchNorm2d
        
        # Layer normalization normalizes each each batch sample separately across all features
        elif normalization_type == 'layernorm':
            normalization_layer = LayerNorm
            
        elif normalization_type == None:
            normalization_layer = Identity
        else:
            sys.exit(f'Wrong value for argument "normalization_type": {normalization_type}')
        
        self.conv_layer_1 = Conv2d(in_channels=conv_1_in_dim, out_channels=conv_1_out_dim, kernel_size=conv_1_kernel_size,
                                   stride=conv_1_stride, padding=conv_1_padding)
        self.normalization_1 = normalization_layer(num_norm_features_1)
        
        self.conv_layer_2 = Conv2d(in_channels=conv_2_in_dim, out_channels=conv_2_out_dim, kernel_size=conv_2_kernel_size,
                                   stride=conv_2_stride, padding=conv_2_padding)
        self.normalization_2 = normalization_layer(num_norm_features_2)
        
        self.conv_layer_3 = Conv2d(in_channels=conv_3_in_dim, out_channels=conv_3_out_dim, kernel_size=conv_3_kernel_size,
                                   stride=conv_3_stride, padding=conv_3_padding)
        self.normalization_3 = normalization_layer(num_norm_features_3)
        
        self.pooling = AvgPool2d(kernel_size=pooling_kernel_size, padding=pooling_zero_padding)
        
        if non_linearity_function == 'relu':
            self.non_linearity = ReLU()
        elif non_linearity_function == 'elu':
            self.non_linearity = ELU()
        elif non_linearity_function == 'gelu':
            self.non_linearity = GELU()
        else:
            sys.exit(f'Wrong value for argument "non_linearity_function": {non_linearity_function}')
        
        self.dropout = Dropout(dropout)
        self.normalization_type = normalization_type


    def forward(self, X):
        
        # X is now of size [batch_size, num_frames, num_channels, frame_len]
        # --> with default values torch.Size([batch_size, 14, 1, 400])
        
        # We convert X into size [batch_size, num_channels, num_frames, frame_len]
        X = X.permute(0, 2, 1, 3)
        
        if self.normalization_type == 'layernorm':
            X = self.dropout(self.non_linearity(self.normalization_1(self.conv_layer_1(X).permute(0, 3, 2, 1)).permute(0, 3, 2, 1)))
            X = self.dropout(self.non_linearity(self.normalization_2(self.conv_layer_2(X).permute(0, 3, 2, 1)).permute(0, 3, 2, 1)))
            X = self.dropout(self.pooling(self.non_linearity(self.normalization_3(self.conv_layer_3(X).permute(0, 3, 2, 1)).permute(0, 3, 2, 1))))
        else:
            X = self.dropout(self.non_linearity(self.normalization_1(self.conv_layer_1(X))))
            X = self.dropout(self.non_linearity(self.normalization_2(self.conv_layer_2(X))))
            X = self.dropout(self.pooling(self.non_linearity(self.normalization_3(self.conv_layer_3(X)))))
        
        X = X.squeeze()
        X = X.permute(0, 2, 1)
        
        # X_output is now of size [batch_size, num_frames, conv_3_out_dim]
        
        return X






class pfml_decoder_linear(Module):
    """
    A linear decoder for the PFML method (one-layer MLP). This decoder is used to convert
    the Transformer outputs into predicted frame-level functionals.
    
    """
    
    def __init__(self,
                 input_dim = 128,
                 output_dim = 11):

        super().__init__()
        
        self.linear_layer = Linear(in_features=input_dim, out_features=output_dim)

    def forward(self, X):
        
        # X is now of size [batch_size, num_frames_embedding, num_features_embedding]
        X_output = self.linear_layer(X)
        
        # X_output is of size [batch_size, num_frames_embedding, output_dim]
        
        return X_output





class absolute_positional_encoding(Module):
    """
    The absolute positional encoding using sinusoids. Code adapted from Harvard's tutorial:
        http://nlp.seas.harvard.edu/annotated-transformer/
    
    The advantage of absolute positional encodings is that, in theory, they can allow the
    model to extrapolate to sequence lengths longer than the ones encountered during training.
    In the original Transformer paper, the authors applied dropout with a rate of 0.1 to the
    sums of the embeddings and the positional encodings as a regularization technique.
    
    """
    
    def __init__(self, d_model=768, max_sequence_length=601, dropout_pos_encoding=0.1):

        super().__init__()
        
        # We apply dropout to the sums of the embeddings and the positional encodings
        self.dropout = Dropout(dropout_pos_encoding)
        
        # We compute positional encodings in the log space
        positional_encoding = torch.zeros(max_sequence_length, d_model)
        position = torch.arange(0, max_sequence_length).unsqueeze(1)
        division_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        # The frequency and offset of the sinusoid is different for each dimension
        positional_encoding[:, 0::2] = torch.sin(position * division_term)
        positional_encoding[:, 1::2] = torch.cos(position * division_term)
        
        # We save positional_encoding as a buffer, i.e. as a parameter in the model which
        # should be saved and restored in the state_dict, but not trained by the optimizer.
        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer("positional_encoding", positional_encoding)
        
    def forward(self, X):
        """
        x: Tensor, shape [batch_size, seq_len, embedding_dim]
        (pos. enc. should be added along seq_len)
        """
        
        X = X + self.positional_encoding[:, :X.size(1), :].requires_grad_(False)
        X = self.dropout(X)
        
        return X


class relative_positional_encoding(Module):
    """
    The relative positional encoding using a convolutional layer, as in the wav2vec 2.0 and
    data2vec papers. As argued by Mohamed et al. (2019) (https://arxiv.org/abs/1904.11660),
    the convolutional positional encoding can bring an advantage over the absolute positional
    encoding in some situations. In wav2vec 2.0 and data2vec, no dropout or learnable convolution
    bias was used, and a layer normalization was applied after the activation function.
    
    """
    def __init__(self, conv_in_dim=768, conv_out_dim=768, conv_kernel_size=3, conv_stride=1,
                 conv_padding=1, conv_bias=False, dropout_pos_encoding=0.0, use_layernorm=True):

        super().__init__()
        
        self.conv = Conv1d(in_channels=conv_in_dim, out_channels=conv_out_dim, kernel_size=conv_kernel_size,
                      stride=conv_stride, padding=conv_padding, bias=conv_bias)
        
        self.non_linearity_gelu = GELU()
        self.layernorm = LayerNorm(conv_out_dim)
        self.dropout = Dropout(dropout_pos_encoding)
        self.use_layernorm = use_layernorm
        
    def forward(self, X):
        # X is of shape [batch_size, num_frames_input, num_features]
        X_pos_conv = self.dropout(self.non_linearity_gelu(self.conv(X.permute(0, 2, 1))))
        
        # We reshape X_pos_conv to shape [batch_size, num_frames_input, num_features] before the addition
        X = X + X_pos_conv.permute(0, 2, 1)
        if self.use_layernorm:
            X = self.layernorm(X)
        
        return X








class pfml_transformer_finetuning(Module):
    """
    The Transformer encoder-based model that is used for fine-tuning PFML pre-trained
    models. The target labels can either be frame-level or sequence-level. This model
    adds two fully-connected layers after the pre-trained model in order to turn the
    Transformer output into categorical probabilities for each output category.
    
    """
    def __init__(self,
                 dim_model = 140,
                 dim_feedforward = 140,
                 classification_layer_latent_dim = 256,
                 output_channels = 7,
                 num_heads = 10,
                 num_encoder_layers = 4,
                 dropout = 0.3,
                 transformer_activation_function = 'gelu',
                 non_linearity_classification_function = 'gelu',
                 num_added_classification_layers = 2,
                 linear_projection_nonlinearity = False,
                 only_attend_to_previous_context = False,
                 use_sqrt = False,
                 use_embedding_projection = True,
                 use_final_projection = True,
                 include_cls_token = False,
                 is_cls_token_random = False,
                 positional_encoding_type = 'absolute',
                 dropout_pos_encoding = 0.0,
                 abs_pos_encoding_max_sequence_length = 260,
                 rel_pos_encoding_conv_in_dim = 140,
                 rel_pos_encoding_conv_out_dim = 140,
                 rel_pos_encoding_conv_kernel_size = 3,
                 rel_pos_encoding_conv_stride = 1,
                 rel_pos_encoding_conv_padding = 1,
                 rel_pos_encoding_conv_bias = False,
                 rel_pos_encoding_use_layernorm = True,
                 sequence_level_classification = False):
        
        super().__init__()
        
        self.include_cls_token = include_cls_token
        if include_cls_token:
            if is_cls_token_random:
                torch.manual_seed(212)
                self.cls_token = torch.Tensor(dim_model).uniform_()
                t = 1000 * time.time() # current time in milliseconds
                torch.manual_seed(int(t) % 2**32)
            else:
                self.cls_token = torch.ones(dim_model)
        
        if positional_encoding_type == 'absolute':
            self.positional_encoder = absolute_positional_encoding(d_model=dim_model,
                                                                   max_sequence_length=abs_pos_encoding_max_sequence_length,
                                                                   dropout_pos_encoding=dropout_pos_encoding)
        elif positional_encoding_type == 'relative':
            self.positional_encoder = relative_positional_encoding(conv_in_dim=rel_pos_encoding_conv_in_dim,
                                                                   conv_out_dim=rel_pos_encoding_conv_out_dim,
                                                                   conv_kernel_size=rel_pos_encoding_conv_kernel_size,
                                                                   conv_stride=rel_pos_encoding_conv_stride,
                                                                   conv_padding=rel_pos_encoding_conv_padding,
                                                                   conv_bias=rel_pos_encoding_conv_bias,
                                                                   dropout_pos_encoding=dropout_pos_encoding,
                                                                   use_layernorm=rel_pos_encoding_use_layernorm)
        else:
            sys.exit("The argument 'positional_encoding_type' should be either 'absolute' or 'relative'")
        
        self.transformer_encoder = Transformer_encoder_base(d_model=dim_model, nhead=num_heads,
                                                            num_encoder_layers=num_encoder_layers,
                                                            dim_feedforward=dim_feedforward, dropout=dropout,
                                                            activation=transformer_activation_function, batch_first=True)
        
        self.embedding_projection = Linear(dim_model, dim_model)
        self.final_projection = Linear(dim_model, dim_model)
        self.dropout = Dropout(dropout)
        
        if num_added_classification_layers == 2:
            self.classification_layer_1 = Linear(dim_model, classification_layer_latent_dim)
            self.classification_layer_2 = Linear(classification_layer_latent_dim, output_channels)
        elif num_added_classification_layers == 1:
            self.classification_layer = Linear(dim_model, output_channels)
        else:
            sys.exit(f'{num_added_classification_layers} is not an option! Options: 1 or 2')
        
        if non_linearity_classification_function == 'elu':
            self.non_linearity_classification = ELU()
        elif non_linearity_classification_function == 'gelu':
            self.non_linearity_classification = GELU()
        else:
            sys.exit(f'{non_linearity_classification_function} is not an option! Options: "elu" or "gelu"')
        
        self.dim_model = dim_model
        self.use_sqrt = use_sqrt
        self.only_attend_to_previous_context = only_attend_to_previous_context
        self.use_embedding_projection = use_embedding_projection
        self.use_final_projection = use_final_projection
        self.num_added_classification_layers = num_added_classification_layers
        self.linear_projection_nonlinearity = linear_projection_nonlinearity
        self.sequence_level_classification = sequence_level_classification
    
    
    def create_src_square_mask(self, sequence_length):
        # Creates a triangular matrix where the elements on the upper triangle are -inf,
        # i.e. the self-attention layers are only allowed to attend to the previous context.
        mask = torch.triu(torch.ones(sequence_length, sequence_length) * float('-inf'), diagonal=1)
        
        return mask
    
        
    def forward(self, src, src_key_padding_mask=None):
        
        if self.only_attend_to_previous_context:
            if self.include_cls_token:
                src_mask = self.create_src_square_mask(src.size()[1] + 1).to(src.device)
            else:
                src_mask = self.create_src_square_mask(src.size()[1]).to(src.device)
        else:
            src_mask = None
        
        if self.use_embedding_projection:
            src = self.embedding_projection(src)
        
        # In the original Transformer paper, the embeddings were multiplied with the square root of the model
        # dimensionality in order to make the positional encodings less dominant
        if self.use_sqrt:
            src = self.positional_encoder(src * math.sqrt(self.dim_model))
        else:
            src = self.positional_encoder(src)
        
        # Add the CLS token to the beginning of the sequence (and also to the beginning of the embedding masks, if necessary)
        if self.include_cls_token:
            src = torch.cat((self.cls_token.repeat(src.size()[0], 1).unsqueeze(1).to(src.device), src), dim=1)
            src_key_padding_mask = torch.cat((torch.from_numpy(np.array([False])).repeat(src.size()[0], 1).to(src.device), src_key_padding_mask), dim=1)
        
        # Transformer blocks - Out size = (batch_size, sequence length, dim_model)
        output, outputs, ff_outputs, ff_residual_outputs = self.transformer_encoder(src, src_mask=src_mask,
                                                                                    src_key_padding_mask=src_key_padding_mask)
        if self.use_final_projection:
            output = self.final_projection(output)
        
        if self.sequence_level_classification:
            # We take the embedding of the first timestep only as we are doing sequence-level classification
            output = output[:, 0, :]
        
        if self.num_added_classification_layers == 2:
            classification_output = self.dropout(self.non_linearity_classification(self.classification_layer_1(output)))
            classification_output = self.non_linearity_classification(self.classification_layer_2(classification_output))
        else:
            if self.linear_projection_nonlinearity:
                classification_output = self.non_linearity_classification(self.classification_layer(output))
            else:
                classification_output = self.classification_layer(output)
        
        return classification_output, outputs, ff_outputs, ff_residual_outputs









class pfml_transformer_encoder(Module):
    """
    The Transformer encoder-based model that is used for PFML pre-training and extracting
    features from PFML pre-trained models.
    
    """
    def __init__(self,
                 dim_model = 128,
                 dim_feedforward = 512,
                 num_heads = 8,
                 num_encoder_layers = 6,
                 dropout = 0.0,
                 transformer_activation_function = 'gelu',
                 require_same_num_embedding_masks = False,
                 prob_frame_is_start_of_embedding_mask = 0.065,
                 embedding_mask_length_frames = 10,
                 min_num_mask_start_frames = 1,
                 learnable_mask_embedding = False,
                 mask_type = 'ones',
                 only_attend_to_previous_context = False,
                 use_sqrt = False,
                 use_embedding_projection = False,
                 use_final_projection = True,
                 include_cls_token = False,
                 is_cls_token_random = False,
                 positional_encoding_type = 'relative',
                 dropout_pos_encoding = 0.0,
                 abs_pos_encoding_max_sequence_length = 301,
                 rel_pos_encoding_conv_in_dim = 128,
                 rel_pos_encoding_conv_out_dim = 128,
                 rel_pos_encoding_conv_kernel_size = 25,
                 rel_pos_encoding_conv_stride = 1,
                 rel_pos_encoding_conv_padding = 12,
                 rel_pos_encoding_conv_bias = False,
                 rel_pos_encoding_use_layernorm = True):
        
        super().__init__()
        
        self.include_cls_token = include_cls_token
        if include_cls_token:
            if is_cls_token_random:
                torch.manual_seed(212)
                self.cls_token = torch.Tensor(dim_model).uniform_()
                t = 1000 * time.time() # current time in milliseconds
                torch.manual_seed(int(t) % 2**32)
            else:
                self.cls_token = torch.ones(dim_model)
        
        if positional_encoding_type == 'absolute':
            self.positional_encoder = absolute_positional_encoding(d_model=dim_model,
                                                                   max_sequence_length=abs_pos_encoding_max_sequence_length,
                                                                   dropout_pos_encoding=dropout_pos_encoding)
        elif positional_encoding_type == 'relative':
            self.positional_encoder = relative_positional_encoding(conv_in_dim=rel_pos_encoding_conv_in_dim,
                                                                   conv_out_dim=rel_pos_encoding_conv_out_dim,
                                                                   conv_kernel_size=rel_pos_encoding_conv_kernel_size,
                                                                   conv_stride=rel_pos_encoding_conv_stride,
                                                                   conv_padding=rel_pos_encoding_conv_padding,
                                                                   conv_bias=rel_pos_encoding_conv_bias,
                                                                   dropout_pos_encoding=dropout_pos_encoding,
                                                                   use_layernorm=rel_pos_encoding_use_layernorm)
        else:
            sys.exit("The argument 'positional_encoding_type' should be either 'absolute' or 'relative'")
        
        self.transformer_encoder = Transformer_encoder_base(d_model=dim_model, nhead=num_heads,
                                                            num_encoder_layers=num_encoder_layers,
                                                            dim_feedforward=dim_feedforward, dropout=dropout,
                                                            activation=transformer_activation_function, batch_first=True)
        
        self.embedding_projection = Linear(dim_model, dim_model)
        self.final_projection = Linear(dim_model, dim_model)
        
        self.dim_model = dim_model
        self.use_sqrt = use_sqrt
        self.only_attend_to_previous_context = only_attend_to_previous_context
        self.use_embedding_projection = use_embedding_projection
        self.use_final_projection = use_final_projection
        self.require_same_num_embedding_masks = require_same_num_embedding_masks
        self.prob_frame_is_start_of_embedding_mask = prob_frame_is_start_of_embedding_mask
        self.embedding_mask_length_frames = embedding_mask_length_frames
        self.min_num_mask_start_frames = min_num_mask_start_frames
        self.learnable_mask_embedding = learnable_mask_embedding
        
        if learnable_mask_embedding:
            self.mask_embedding = Parameter(torch.Tensor(dim_model).uniform_())
        else:
            if mask_type == 'random':
                torch.manual_seed(222)
                self.mask_embedding = torch.Tensor(dim_model).uniform_()
                t = 1000 * time.time() # current time in milliseconds
                torch.manual_seed(int(t) % 2**32)
            elif mask_type == 'ones':
                self.mask_embedding = torch.ones(dim_model)
            elif mask_type == 'zeros':
                self.mask_embedding = torch.zeros(dim_model)
            else:
                sys.exit(f'{mask_type} is not a valid option for the argument "mask_type"!')
    
    
    def compute_embedding_mask_indices(self, batch_size, num_frames, embedding_mask_frame_start_prob,
                                       mask_length, min_mask_start_frames, same_num_masks, padding_masks):
        
        indices_embedding_masks_initial = []
        num_embedding_masks_initial = []
        
        # We go through each element in the batch and create initial masks
        for i in range(batch_size):
            padding_mask = padding_masks[i, :]
            
            # A boolean array of the size of num_frames that will contain the indices of masked frames
            indices_embedding_mask_initial = np.full(num_frames, False)
            
            # We do not let the embedding mask get over the padding mask
            max_possible_embedding_mask_start_index = num_frames - len(padding_mask[padding_mask == True]) - mask_length + 1
            embedding_mask_start = np.random.uniform(size=max_possible_embedding_mask_start_index) < embedding_mask_frame_start_prob
            
            # We make sure that we have at least min_mask_start_frames of start indices for the embedding masks
            while len(embedding_mask_start[embedding_mask_start == True]) < min_mask_start_frames:
                embedding_mask_start = np.random.uniform(size=max_possible_embedding_mask_start_index) < embedding_mask_frame_start_prob
            
            # We make mask spans of length mask_length, starting from the indices indices_mask_start
            indices_mask_start = np.where(embedding_mask_start == True)[0]
            for j in range(len(indices_mask_start)):
                mask_start_index = indices_mask_start[j]
                indices_embedding_mask_initial[mask_start_index:(mask_start_index + mask_length)] = True
            
            indices_embedding_masks_initial.append(indices_embedding_mask_initial)
            
            if same_num_masks:
                # The number of mask start frames in the non-padded segment
                num_embedding_masks_initial.append(len(indices_embedding_mask_initial[indices_embedding_mask_initial == True]))
        
        if same_num_masks:
            num_embedding_masks_initial = np.array(num_embedding_masks_initial)
            indices_embedding_masks = []
            
            # We find out the minimum number of embedding mask indices (we want to have same number of masks
            # in each batch item)
            min_num_embedding_mask_frames = np.amin(num_embedding_masks_initial)
            for i in range(batch_size):
                indices_embedding_mask = indices_embedding_masks_initial[i]
                indices_mask = np.where(indices_embedding_mask == True)[0]
                if len(indices_mask) > min_num_embedding_mask_frames:
                    # We have too many mask start indices, so we remove n indices so that we get the same number
                    # of embedding masks in each batch item
                    num_removed_masks = len(indices_mask) - min_num_embedding_mask_frames
                    indices_removed_mask = np.random.choice(indices_mask, num_removed_masks, replace=False)
                    indices_embedding_mask[indices_removed_mask] = False
                indices_embedding_masks.append(indices_embedding_mask)
            
            return np.array(indices_embedding_masks)
        else:
            return np.array(indices_embedding_masks_initial)
    
    
    def create_src_square_mask(self, sequence_length):
        # Creates a triangular matrix where the elements on the upper triangle are -inf,
        # i.e. the self-attention layers are only allowed to attend to the previous context.
        mask = torch.triu(torch.ones(sequence_length, sequence_length) * float('-inf'), diagonal=1)
        
        return mask
    
        
    def forward(self, src, src_key_padding_mask=None, mask_embeddings=False, output_type=None):
        
        target_output_types = [None, 'ff_outputs', 'ff_residual_outputs', 'end_of_block',
                               'ff_output_second_last', 'ff_residual_output_second_last', 'end_of_block_second_last']
        if output_type not in target_output_types:
            sys.exit(f'The argument "output_type" should be one of the following: {target_output_types}')
        
        if self.only_attend_to_previous_context:
            if self.include_cls_token:
                src_mask = self.create_src_square_mask(src.size()[1] + 1).to(src.device)
            else:
                src_mask = self.create_src_square_mask(src.size()[1]).to(src.device)
        else:
            src_mask = None
        
        if self.use_embedding_projection:
            src = self.embedding_projection(src)
        
        # Apply an embedding mask
        if mask_embeddings:
            indices_embedding_masks = self.compute_embedding_mask_indices(src.size()[0], src.size()[1],
                                                                          self.prob_frame_is_start_of_embedding_mask,
                                                                          self.embedding_mask_length_frames,
                                                                          self.min_num_mask_start_frames,
                                                                          self.require_same_num_embedding_masks,
                                                                          src_key_padding_mask.cpu().numpy())
            
            for i in range(src.size()[0]):
                if self.learnable_mask_embedding:
                    src[i, indices_embedding_masks[i], :] = self.mask_embedding
                else:
                    src[i, indices_embedding_masks[i], :] = self.mask_embedding.to(src.device)
        
        # Apply positional encoding. In the original Transformer paper, the embeddings were multiplied
        # with the square root of the model dimensionality in order to make the positional encodings less dominant
        if self.use_sqrt:
            src = self.positional_encoder(src * math.sqrt(self.dim_model))
        else:
            src = self.positional_encoder(src)
        
        # Add the CLS token to the beginning of the sequence (and also to the beginning of the embedding masks, if necessary)
        if self.include_cls_token:
            src = torch.cat((self.cls_token.repeat(src.size()[0], 1).unsqueeze(1).to(src.device), src), dim=1)
            src_key_padding_mask = torch.cat((torch.from_numpy(np.array([False])).repeat(src.size()[0], 1).to(src.device), src_key_padding_mask), dim=1)
            if mask_embeddings:
                indices_embedding_masks = np.concatenate((np.expand_dims(np.repeat(False, src.size()[0]), axis=1), indices_embedding_masks), axis=1)
        
        # Transformer blocks - Out size = (batch_size, sequence length, dim_model)
        output, outputs, ff_outputs, ff_residual_outputs = self.transformer_encoder(src, src_mask=src_mask,
                                                                                    src_key_padding_mask=src_key_padding_mask)
        
        if self.use_final_projection:
            output = self.final_projection(output)
        
        if output_type == None:
            if mask_embeddings:
                return output, torch.from_numpy(indices_embedding_masks).to(src.device)
            else:
                return output
        elif output_type == 'ff_output_second_last':
            return ff_outputs[-2]
        elif output_type == 'ff_residual_output_second_last':
            return ff_residual_outputs[-2]
        elif output_type == 'end_of_block_second_last':
            return outputs[-2]
        else:
            if output_type == 'ff_outputs':
                output_unaveraged = ff_outputs
            elif output_type == 'ff_residual_outputs':
                output_unaveraged = ff_residual_outputs
            elif output_type == 'end_of_block':
                output_unaveraged = outputs
            output_averaged = sum(output_unaveraged) / len(output_unaveraged)
            
            return output_averaged
    