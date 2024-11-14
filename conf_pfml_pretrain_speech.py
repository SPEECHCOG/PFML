#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Einari Vaaras, einari.vaaras@tuni.fi, Tampere University
Speech and Cognition Research Group, https://webpages.tuni.fi/specog/index.html

The configuration file for pfml_pretrain_speech.py.

"""

experiment_num = 1

"""The hyperparameters for our training and feature extraction processes"""

# The maximum number of training epochs
max_epochs = 10000

# The patience counter for early stopping
patience = 25

# Dropout rate of the encoder model
dropout_encoder = 0.2

# The minibatch size
batch_size = 64

# The learning rate of our model training
learning_rate = 1e-4

# Flag for running PFML pre-training
train_model = True

# Flag for using our PFML pre-trained model to extract features
extract_features = False

# Flag for loading the weights for our model, i.e. flag for continuing a previous training process
load_model = False

# Flag for saving the best model (according to validation loss) after each training epoch where the
# validation loss is lower than before
save_best_model = True

# The name of the text file into which we log the output of the training process
name_of_log_textfile = f'trainlogs_pfml_pretraining/pfml_trainlog_librispeech_{experiment_num}.txt'

# A flag for determining whether we want to print the contents of the configuration file into the
# logging file
print_conf_contents = True

# Define our models that we want to use from the file pfml_model.py
encoder_name = 'raw_audio_encoder_CNN'
transformer_name = 'pfml_transformer_encoder'
decoder_name = 'pfml_decoder_linear'

# Define our dataset for our data loader that we want to use from the file pfml_data_loader.py
dataset_name = 'pfml_raw_audio_dataset_librispeech'

# The window length in seconds (for the signal framing)
window_len_seconds = 0.03

# The hop length in seconds (for the signal framing)
hop_len_seconds = 0.01

# The target sampling rate (in Hz) that we want to use (if the sampling rate of the data is not the same as
# target_fs, the dataloader will perform resampling)
target_fs = 16000

# Defines the minimum number of training epochs, just to make sure that we don't stop training too soon if
# the variance of the embeddings is too low in the beginning of the training process
min_train_epochs = 20

# The minimum acceptable variances of our embeddings. If we don't care about the minimum acceptable
# variance, we can set this variable to some negative value (variance cannot be less than 0)
min_embedding_variance = -9999.0

# A flag for computing the prediction loss only for the masked embeddings (True). If set to False,
# the loss is computed for the non-padded parts of the embeddings, including the non-masked parts.
compute_loss_only_for_masked_embeddings = True

# Define our loss function that we want to use from torch.nn
pfml_loss_name = 'L1Loss'

# The scaling multipliers for the loss functions, a value of 1.0 means no scaling
pfml_loss_scaler = 1.0

# The hyperparameters for the loss functions
pfml_loss_params = {}

# A flag for defining whether we want to compute the variance over the time dimension only for non-padded
# and unmasked parts of our predicted outputs and training targets (True), or only for the non-padded
# parts (False). Since the embedding masks can bring additional variance to the outputs, it might be a good
# choice to ignore them when computing the variance.
compute_variance_for_unmasked_parts = True

# Define the optimization algorithm we want to use from torch.optim
optimization_algorithm = 'RAdam'

# The hyperparameters for our optimization algorithm
optimization_algorithm_params = {'lr': learning_rate}

# A flag to determine if we want to use a learning rate scheduler
use_lr_scheduler = True

# Define which learning rate scheduler we want to use from torch.optim.lr_scheduler
lr_scheduler = 'ReduceLROnPlateau'

# The hyperparameters for the learning rate scheduler
lr_scheduler_params = {'mode': 'min',
                       'factor': 0.5,
                       'patience': 10}

# The names of the model weight files of the best models (according to validation loss) for
# loading/saving model weights
encoder_best_model_name = f'pfml_pretrained_models/pfml_Encoder_speech_best_model_{experiment_num}.pt'
transformer_best_model_name = f'pfml_pretrained_models/pfml_Transformer_speech_best_model_{experiment_num}.pt'
decoder_best_model_name = f'pfml_pretrained_models/pfml_Decoder_speech_best_model_{experiment_num}.pt'

# The base name of the files containing the output of the feature extraction process
feature_extraction_model_output_savefile_basename = f'feature_extraction_output/librispeech_pfml_feats_{experiment_num}'




"""The hyperparameters for our Transformer encoder"""
# The dimensionality of the input embedding sequences for the Transformer encoder
embedding_dim = 128

# The size of the hidden dimension of the feed-forward neural network part of the Transformer encoder blocks
transformer_hidden_dim = 512

# The number of attention heads for each multi-head self-attention
num_attention_heads = 8

# The number of Transformer encoder blocks
num_transformer_encoder_layers = 6

# The dropout of the Transformer encoder blocks
dropout_transformer = 0.2

# The activation function for the Transformer feed-forward neural network part. Options: 'relu' and 'gelu'.
# ReLU was used in the original Transformer paper, whereas GELU was used in e.g. wav2vec 2.0 and data2vec.
transformer_activation_function = 'gelu'

# Defines whether we want to have the same number of embedding masks in each batch element (as was used in e.g. the
# original data2vec implementation in Fairseq). If set to True: After computing the embedding mask indices, the
# minimum number of embedding masks in a batch element is first defined. Then, for the rest of the batch elements
# containing more embedding masks, mask indices are randomly removed until each batch element has the same number
# of embedding masks. Please note that this might be problematic if there are large differences between the lengths
# of the batch elements (e.g. a long sample might have very few masks compared to the length of the sample).
require_same_num_embedding_masks = False

# The probability of a frame being the start of an embedding mask when masking embeddings
# for the student network
prob_frame_is_start_of_embedding_mask = 0.065

# The length of the embedding masks (in frames) when masking the embeddings for the student network
embedding_mask_length_frames = 10

# The minimum number of embedding mask starting frames in each embedding (the embedding mask start indices are
# chosen randomly, so without this parameter there is a chance that there might be # no masked frames at all
min_num_mask_start_frames = 1

# Defines whether we want to use a learnable mask embedding (as in e.g. the data2vec paper). If set to False,
# the masked parts of the embeddings are replaced with a mask token (see next hyperparameter)
learnable_mask_embedding = False

# Defines the type of the mask token. Options: 'random' / 'ones' / 'zeros'. This hyperparameter is neglected
# if learnable_mask_embedding = True
mask_type = 'ones'

# Defines what output of the Transformer encoder blocks we want to use as our training targets (the targets are
# instance-normalized and averaged, except for the option None). There are three possible options, all present in 
# the data2vec paper (Table 4):
#     None: The output of the last Transformer encoder block, without instance-normalizing or averaging
#    'ff_outputs': The output of the feed-forward (FFN) part of the Transformer encoder
#    'ff_residual_outputs': The output of the FFN of the Transformer encoder after adding the residual
#    'end_of_block': The output of the FFN of the Transformer encoder after the residual connection and LayerNorm
target_output_type = None

# Defines whether our Transformer encoder is bidirectional (False) or left-to-right (True). In e.g. data2vec
# and BERT, a bidirectional version was used.
only_attend_to_previous_context = False

# Defines whether we want to multiply the embeddings with the square root of the model dimensionality
# before we compute the positional encodings. In the original Transformer paper, this was done to make
# the positional encodings less dominant compared to the embeddings.
use_sqrt = False

# Defines whether we want to apply a linear projection to the embeddings after the positional encoding.
use_embedding_projection = False

# Defines whether we want to apply a linear projection after the final Transformer encoder block
use_final_projection = True

# Defines whether we want to use absolute positional encodings (using sinusoids) or relative positional
# encodings (using a CNN layer) for our embeddings. Relative positional encoding was used in e.g. the
# data2vec paper, whereas absolute positional encodings were used in the original Transformer paper.
#     Options: 'absolute' or 'relative'
positional_encoding_type = 'relative'

# Defines the dropout of our positional encodings (applies to both the absolute and relative positional
# encodings). In the original Transformer paper, a dropout of 0.1 was used as a regularization technique
dropout_pos_encoding = 0.0

# (Only related to absolute positional encodings) Defines the maximum sequence length in frames
abs_pos_encoding_max_sequence_length = 301

# (Only related to relative positional encodings)
rel_pos_encoding_conv_in_dim = embedding_dim # The input dimensionality of the positional encodings
rel_pos_encoding_conv_out_dim = embedding_dim # The output dimensionality of the positional encodings
rel_pos_encoding_conv_kernel_size = 25 # The CNN kernel size of the positional encodings
rel_pos_encoding_conv_stride = 1 # The CNN stride of the positional encodings
rel_pos_encoding_conv_padding = 12 # The CNN padding of the positional encodings
rel_pos_encoding_conv_bias = False # The CNN bias of the pos. encodings (not used in wav2vec 2.0 and data2vec papers)
rel_pos_encoding_use_layernorm = True # Defines whether we want to apply LayerNorm after the positional encoding



"""Other hyperparameters"""
# The hyperparameters for constructing the encoder model. Empty dictionary = use default hyperparameters
encoder_model_params = {'conv_1_in_dim': 1,
                        'conv_1_out_dim': 128,
                        'num_norm_features_1': 128,
                        'conv_2_in_dim': 128,
                        'conv_2_out_dim': 128,
                        'num_norm_features_2': 128,
                        'conv_3_in_dim': 128,
                        'conv_3_out_dim': 128,
                        'num_norm_features_3': 128,
                        'conv_4_in_dim': 128,
                        'conv_4_out_dim': 128,
                        'num_norm_features_4': 128,
                        'dropout': dropout_encoder}

# The hyperparameters for constructing the Transformer model. Empty dictionary = use default hyperparameters
transformer_params = {'dim_model': embedding_dim,
                      'dim_feedforward': transformer_hidden_dim,
                      'num_heads': num_attention_heads,
                      'num_encoder_layers': num_transformer_encoder_layers,
                      'dropout': dropout_transformer,
                      'transformer_activation_function': transformer_activation_function,
                      'require_same_num_embedding_masks': require_same_num_embedding_masks,
                      'prob_frame_is_start_of_embedding_mask': prob_frame_is_start_of_embedding_mask,
                      'embedding_mask_length_frames': embedding_mask_length_frames,
                      'min_num_mask_start_frames': min_num_mask_start_frames,
                      'learnable_mask_embedding': learnable_mask_embedding,
                      'mask_type': mask_type,
                      'only_attend_to_previous_context': only_attend_to_previous_context,
                      'use_sqrt': use_sqrt,
                      'use_embedding_projection': use_embedding_projection,
                      'use_final_projection': use_final_projection,
                      'positional_encoding_type': positional_encoding_type,
                      'dropout_pos_encoding': dropout_pos_encoding,
                      'abs_pos_encoding_max_sequence_length': abs_pos_encoding_max_sequence_length,
                      'rel_pos_encoding_conv_in_dim': rel_pos_encoding_conv_in_dim,
                      'rel_pos_encoding_conv_out_dim': rel_pos_encoding_conv_out_dim,
                      'rel_pos_encoding_conv_kernel_size': rel_pos_encoding_conv_kernel_size,
                      'rel_pos_encoding_conv_stride': rel_pos_encoding_conv_stride,
                      'rel_pos_encoding_conv_padding': rel_pos_encoding_conv_padding,
                      'rel_pos_encoding_conv_bias': rel_pos_encoding_conv_bias,
                      'rel_pos_encoding_use_layernorm': rel_pos_encoding_use_layernorm}

# The hyperparameters of our decoder model. Empty dictionary = use default hyperparameters
decoder_params = {'input_dim': 128,
                  'output_dim': 11}

# The hyperparameters for our data loaders
params_train_dataset = {'max_length_seconds': 3.0,
                        'window_len_seconds': window_len_seconds,
                        'hop_len_seconds': hop_len_seconds,
                        'target_fs': target_fs}
params_validation_dataset = {'max_length_seconds': 3.0,
                             'window_len_seconds': window_len_seconds,
                             'hop_len_seconds': hop_len_seconds,
                             'target_fs': target_fs}
params_feature_extraction_dataset = {'max_length_seconds': 3.0,
                                     'window_len_seconds': window_len_seconds,
                                     'hop_len_seconds': hop_len_seconds,
                                     'target_fs': target_fs}

# The hyperparameters for training and validation (arguments for torch.utils.data.DataLoader object)
params_train = {'batch_size': batch_size,
                'shuffle': True,
                'drop_last': False}

# The hyperparameters for using our trained data2vec model to extract features (arguments for
# torch.utils.data.DataLoader object)
params_feature_extraction = {'batch_size': batch_size,
                             'shuffle': False,
                             'drop_last': False}