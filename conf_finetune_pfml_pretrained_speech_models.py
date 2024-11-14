#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Einari Vaaras, einari.vaaras@tuni.fi, Tampere University
Speech and Cognition Research Group, https://webpages.tuni.fi/specog/index.html

The configuration file for finetune_pfml_pretrained_speech_models.py.

"""

finetuning_experiment_number = 1

# The .py configuration file that was used for PFML pre-training (contains all necessary model
# hyperparameters). Note: Do NOT add the '.py' ending to the file name!
pretrained_model_conf_file = 'conf_pfml_pretrain_speech'

# The name of the time-series model that we want to use from the file pfml_model.py for fine-tuning
timeseries_model = 'pfml_transformer_finetuning'

# A flag for determining whether we want to print the contents of the configuration file into the
# logging file
print_conf_contents = True

# The directory where the experimental results are saved (the models and the logging output)
result_dir = f'pfml_finetuning_results_speech_{finetuning_experiment_number}'

# The name of the text file into which we log the output of the training process. Please note that this
# file (and its directory) will be saved under the directory result_dir.
name_of_log_textfile = f'pfml_finetuning_speech_trainlog_{finetuning_experiment_number}.txt'


"""The hyperparameters for our training process"""

# Flag for training our model(s)
train_model = True

# Flag for testing our trained model(s)
test_model = True

# Flag for fine-tuning the Transformer model block by block. If set to False, the entire Transformer model
# will be fine-tuned. If set to True, the Transformer model will be fine-tuned N times, where N is the
# number of Transformer encoder blocks in the model. In each fine-tuning iteration (except the first),
# one Transformer encoder block will be removed from the end of the model, and the rest of the model will
# be fine-tuned. For example, if N = 3, this means that we fine-tune the model three times: First, we
# fine-tune the entire model. Then, we remove one Transformer encoder block and we fine-tune the Transformer
# consisting of 2 encoder blocks. Finally, we remove another block, and we fine-tune a Transformer consisting
# of only one encoder block.
finetune_transformer_block_by_block = False

# The maximum number of training epochs
max_epochs = 800

# The learning rate of our model training
learning_rate = 4e-5

# The number of input sequences that we feed into our model before computing the mean loss (and performing backpropagation
# during training).
batch_size = 16

# The patience counter for early stopping
patience = 50

# Dropout rate of the encoder model
dropout_encoder_model = 0.3

# Dropout rate of the timeseries model
dropout_timeseries_model = 0.3

# Select the training criterion
train_criterion = 'uar' # Options: 'f1' / 'recall' / 'uar'

# The number of folds for k-folds cross-validation
num_folds = 10

# A flag whether we want to randomize the order of the babies before applying k-folds cross-validation
randomize_order_kfolds = True

# A flag to whether we want to use class weighting for our loss
use_class_weights = True

# Define our loss function that we want to use from torch.nn
loss_name = 'CrossEntropyLoss'

# The hyperparameters for the loss function
loss_params = {}

# Define the optimization algorithm we want to use from torch.optim
optimization_algorithm = 'Adam'

# The hyperparameters for our optimization algorithm
optimization_algorithm_params = {'lr': learning_rate}

# A flag to determine if we want to use a learning rate scheduler
use_lr_scheduler = True

# Define our learning rate schedulers for the fine-tuning stages 1 and 2 (from torch.optim.lr_scheduler)
lr_scheduler_stage_1 = 'ReduceLROnPlateau'
lr_scheduler_params_stage_1 = {'mode': 'max',
                               'factor': 0.5,
                               'patience': 15}

lr_scheduler_stage_2_part_1_epochs = 20
lr_scheduler_stage_2_part_1 = 'LinearLR'
lr_scheduler_params_stage_2_part_1 = {'start_factor': 0.001,
                                      'total_iters': lr_scheduler_stage_2_part_1_epochs}
lr_scheduler_stage_2_part_2 = 'ReduceLROnPlateau'
lr_scheduler_params_stage_2_part_2 = {'mode': 'max',
                                      'factor': 0.5,
                                      'patience': 15}


"""Additional hyperparameters for our models"""
additional_hyperparameters_timeseries_model = {'sequence_level_classification': True}


"""The hyperparameters for our dataset and data loaders"""

# The number of randomly generated utterances
num_randomly_generated_utterances = 5000

# The maximum length of the utterances in seconds
max_length_seconds = 3.0

# The sampling rate of the utterances
fs = 16000

# Define our dataset for our data loader that we want to use from the file pfml_data_loader.py
dataset_name = 'random_speech_data_dataset'

# The ratio in which we split our training data into training and validation sets. For example, a ratio
# of 0.8 will result in 80% of our training data being in the training set and 20% in the validation set.
train_val_ratio = 0.8

# Select whether we want to shuffle our training data
shuffle_training_data = True

# The hyperparameters for our data loaders
params_train_dataset = {'max_length_seconds': max_length_seconds,
                        'train_val_ratio': train_val_ratio,
                        'window_len_seconds': 0.03,
                        'hop_len_seconds': 0.01,
                        'fs': fs,
                        'include_artificial_labels': True}

params_validation_dataset = {'max_length_seconds': max_length_seconds,
                             'train_val_ratio': train_val_ratio,
                             'window_len_seconds': 0.03,
                             'hop_len_seconds': 0.01,
                             'fs': fs,
                             'include_artificial_labels': True}

params_test_dataset = {'max_length_seconds': max_length_seconds,
                       'window_len_seconds': 0.03,
                       'hop_len_seconds': 0.01,
                       'fs': fs,
                       'include_artificial_labels': True}

# The hyperparameters for training and validation (arguments for torch.utils.data.DataLoader object)
params_train = {'batch_size': batch_size,
                'shuffle': shuffle_training_data,
                'drop_last': False}

# The hyperparameters for using our trained data2vec model to extract features (arguments for
# torch.utils.data.DataLoader object)
params_test = {'batch_size': batch_size,
               'shuffle': False,
               'drop_last': False}
