# -*- coding: utf-8 -*-
"""
Author: Einari Vaaras, einari.vaaras@tuni.fi, Tampere University
Speech and Cognition Research Group, https://webpages.tuni.fi/specog/index.html

The data loaders for PFML pre-training and fine-tuning for three different data
modalities (speech data, multi-sensor IMU data, and EEG data). The data augmentation
scripts for multi-sensor IMU data have been implemented by Manu Airaksinen.

NOTE: For detailed descriptions regarding the input variables for the data loaders,
see the configuration files.

"""

import numpy as np
from torch.utils.data import Dataset
import os
import sys
import librosa
import scipy




class pfml_raw_audio_dataset_librispeech(Dataset):
    """
    Dataloader for PFML pre-training using the Librispeech (https://www.openslr.org/12) dataset.
    
    """

    def __init__(self, train_val_test='train', max_length_seconds=3.0, train_val_ratio=0.8, random_seed=22,
                 file_dir='./LibriSpeech', normalize_waveform=True, window_len_seconds=0.03, hop_len_seconds=0.01,
                 target_fs=16000, apply_smooth_windowing=False, normalize_functionals_sample_level=False,
                 normalize_functionals_corpus_level=True, functionals_include_mean=True,
                 functionals_include_var=True, functionals_include_skew=True, functionals_include_kurtosis=True,
                 functionals_include_min=True, functionals_include_max=True, functionals_include_zcr=True,
                 functionals_include_acf_mean=True, functionals_include_acf_var=True,
                 functionals_include_acf_skew=True, functionals_include_acf_kurtosis=True,
                 preprocess_data=False, preprocessed_data_dir='./preprocessed_librispeech_files_framed',
                 precompute_functionals=False, functionals_save_dir='./precomputed_librispeech_functionals'):
        super().__init__()
        
        # Since the utterances contain different numbers of frames, we concatenate the frames of all utterances into
        # one memory-mapped array: frame_offsets stores where each utterance begins and ends in this array.
        frames_path = os.path.join(preprocessed_data_dir, 'frames_librispeech.npy')
        frame_offsets_path = os.path.join(preprocessed_data_dir, 'frame_offsets_librispeech.npy')
        functionals_path = os.path.join(functionals_save_dir, 'functionals_librispeech.npy')
        
        # Find out our FLAC files in the given directory
        try:
            # This is used to spot nonexisting directories since os.walk() is silent about them
            os.listdir(file_dir)
            
            filenames_flac = []
            for dir_path, dir_names, file_names in os.walk(file_dir):
                if len(file_names) > 0:
                    for file_name in file_names:
                        filenames_flac.append(os.path.join(dir_path, file_name))
        except FileNotFoundError:
            sys.exit(f'Given .flac file directory {file_dir} does not exist!')
        
        # Clean the list if there are other files than .flac files
        flac_file_names = [filename for filename in filenames_flac if filename.endswith('.flac')]
        flac_file_names = sorted(flac_file_names, key=lambda x: (int(x.split(os.sep)[-1].split('.')[0].split('-')[0]),
                                                                 int(x.split(os.sep)[-1].split('.')[0].split('-')[1]),
                                                                 int(x.split(os.sep)[-1].split('.')[0].split('-')[2])))
        flac_file_names = np.array(flac_file_names)
        del filenames_flac
        
        # The frame and hop lengths (in samples)
        frame_len = int(window_len_seconds * target_fs)
        shift = int(hop_len_seconds * target_fs)
        
        # Preprocess and save the framed signals
        if not os.path.exists(preprocessed_data_dir):
            os.makedirs(preprocessed_data_dir)
            preprocess_data = True
        else:
            if not os.path.exists(frames_path) or not os.path.exists(frame_offsets_path):
                preprocess_data = True
            if preprocess_data and len(os.listdir(preprocessed_data_dir)) != 0:
                # Remove old files from the given directory
                filenames_old_files = os.listdir(preprocessed_data_dir)
                for filename in filenames_old_files:
                    os.remove(os.path.join(preprocessed_data_dir, filename))
        
        if preprocess_data or len(os.listdir(preprocessed_data_dir)) == 0:
            
            # We first determine the total number of frames without loading all signals into memory
            frame_offsets = np.zeros(len(flac_file_names) + 1, dtype=np.int64)
            for i, filename in enumerate(flac_file_names):
                x, _ = librosa.core.load(filename, sr=target_fs)
                frame_offsets[i + 1] = frame_offsets[i] + int(np.floor(((len(x) - frame_len) / shift) + 1))
            
            # Pre-allocate one array containing the frames of all samples
            feats = np.lib.format.open_memmap(frames_path, mode='w+', dtype=np.float32, shape=(frame_offsets[-1], frame_len))
            np.save(frame_offsets_path, frame_offsets)
            
            # We go through each audio file and write its framed signal into the pre-allocated array
            for i, filename in enumerate(flac_file_names):
                
                x, _ = librosa.core.load(filename, sr=target_fs)
                
                # Normalize to zero mean, unit variance
                if normalize_waveform:
                    x = (x - x.mean()) / x.std()
                
                # We frame the signal. x_framed is of size [num_frames, frame_len]
                x_framed = librosa.util.frame(x, frame_length=frame_len, hop_length=shift, axis=0)
                
                if apply_smooth_windowing:
                    # We apply a Hann window for our frames
                    window = scipy.signal.hann(frame_len, sym=False)
                    x_framed_windowed = np.zeros_like(x_framed)
                    for j in range(x_framed.shape[0]):
                        x_framed_windowed[j,:] = x_framed[j,:] * window
                    x_framed = x_framed_windowed
                
                feats[frame_offsets[i]:frame_offsets[i + 1]] = x_framed
            
            # A memory-mapped array writes data into a file, but we ask the OS to write any pending changes to the
            # file just in case (the OS may temporarily keep some changes in RAM before writing them to disk).
            feats.flush()
            
            del feats
        
        
        # Load the framed signals (memory-mapped array --> not loading the complete array into RAM) and their offsets
        self.feats = np.load(frames_path, mmap_mode='r')
        self.frame_offsets = np.load(frame_offsets_path)
        
        # We define the longest sample length (in frames)
        x_zeros = np.zeros(int(max_length_seconds*target_fs))
        self.x_zeros_framed = librosa.util.frame(x_zeros, frame_length=frame_len, hop_length=shift, axis=0)
        self.longest_sample_length = len(self.x_zeros_framed)
        
        # We compute and save functionals of the features
        if not os.path.exists(functionals_save_dir):
            os.makedirs(functionals_save_dir)
            precompute_functionals = True
        else:
            if not os.path.exists(functionals_path):
                precompute_functionals = True
            if precompute_functionals and len(os.listdir(functionals_save_dir)) != 0:
                # Remove old files from the given directory
                filenames_old_files = os.listdir(functionals_save_dir)
                for filename in filenames_old_files:
                    os.remove(os.path.join(functionals_save_dir, filename))
        
        if precompute_functionals or len(os.listdir(functionals_save_dir)) == 0:
            feats_functionals = None
            for i in range(len(flac_file_names)):
                feat = self.feats[self.frame_offsets[i]:self.frame_offsets[i + 1]]
                functionals = []
                if functionals_include_mean:
                    functionals.append(np.mean(feat, axis=1))
                if functionals_include_var:
                    functionals.append(np.var(feat, axis=1))
                if functionals_include_skew:
                    functionals.append(scipy.stats.skew(feat, axis=1))
                if functionals_include_kurtosis:
                    functionals.append(scipy.stats.kurtosis(feat, axis=1))
                if functionals_include_min:
                    functionals.append(feat.min(axis=1))
                if functionals_include_max:
                    functionals.append(feat.max(axis=1))
                if functionals_include_zcr:
                    functionals.append(librosa.zero_crossings(feat, axis=1).sum(axis=1) / frame_len)
                if functionals_include_acf_mean or functionals_include_acf_var or functionals_include_acf_skew or functionals_include_acf_kurtosis:
                    ac = estimated_autocorrelation(feat)
                    if functionals_include_acf_mean:
                        functionals.append(np.mean(ac, axis=1))
                    if functionals_include_acf_var:
                        functionals.append(np.var(ac, axis=1))
                    if functionals_include_acf_skew:
                        functionals.append(scipy.stats.skew(ac, axis=1))
                    if functionals_include_acf_kurtosis:
                        functionals.append(scipy.stats.kurtosis(ac, axis=1))
                functionals = np.stack(functionals, axis=1)
                if normalize_functionals_sample_level:
                    functionals = normalize_sample(functionals)
                if feats_functionals is None:
                    feats_functionals = np.lib.format.open_memmap(functionals_path, mode='w+', dtype=functionals.dtype, shape=(len(self.feats), functionals.shape[1]))
                feats_functionals[self.frame_offsets[i]:self.frame_offsets[i + 1]] = functionals
            
            if normalize_functionals_corpus_level:
                feats_functionals = normalize_dataset(feats_functionals, self.frame_offsets)
            feats_functionals.flush()
            del feats_functionals
        
        self.feats_functionals = np.load(functionals_path, mmap_mode='r')
        
        # Split our data into a train, validation, and test set
        np.random.seed(random_seed)
        mask_trainval_split = np.random.rand(len(flac_file_names)) <= train_val_ratio
        
        # train_val_test has three options: 'train', 'validation' and 'test'. We use 'test' when we want to extract
        # features using a PFML pre-trained model, i.e. we use all of our data with the option 'test'.
        if train_val_test == 'train':
            self.indices = np.arange(len(flac_file_names))[mask_trainval_split]
        elif train_val_test == 'validation':
            self.indices = np.arange(len(flac_file_names))[~mask_trainval_split]
        else:
            self.indices = np.arange(len(flac_file_names))
        
        self.train_val_test = train_val_test

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index):
        
        data_index = self.indices[index]
        framed_signal_orig = np.array(self.feats[self.frame_offsets[data_index]:self.frame_offsets[data_index + 1]], copy=True)
        functionals = np.array(self.feats_functionals[self.frame_offsets[data_index]:self.frame_offsets[data_index + 1]], copy=True)
        
        # If our sample is shorter than the longest acceptable sample, we add a zero-padded part to the end
        if len(framed_signal_orig) < self.longest_sample_length:
            num_missing_frames = self.longest_sample_length - len(framed_signal_orig)
            framed_signal = np.concatenate((framed_signal_orig, self.x_zeros_framed[:num_missing_frames, :]))
            num_zero_padded_frames = num_missing_frames
            functionals_zeropad = np.zeros((len(framed_signal), functionals.shape[1]))
            functionals = np.concatenate((functionals, functionals_zeropad[:num_zero_padded_frames, :]))
        
        # If our sample is longer than the longest acceptable sample, we take a random segment of the same
        # length as the longest acceptable sample length
        elif len(framed_signal_orig) > self.longest_sample_length:
            if self.train_val_test == 'test':
                np.random.seed(12)
            part_index = np.random.randint(len(framed_signal_orig) - self.longest_sample_length + 1)
            framed_signal = framed_signal_orig[part_index:(part_index + self.longest_sample_length)]
            num_zero_padded_frames = 0
            functionals = functionals[part_index:(part_index + self.longest_sample_length)]
        else:
            framed_signal = framed_signal_orig
            num_zero_padded_frames = 0
        
        # The indices of zero padded frames are tagged with True, whereas non-padded frames are tagged with False
        zero_padding_mask = np.full(len(framed_signal), False)
        if num_zero_padded_frames != 0:
            zero_padding_mask[-num_zero_padded_frames:] = True
        
        return framed_signal, zero_padding_mask, functionals



class random_imu_data_dataset(Dataset):
    """
    Dataloader for PFML pre-training and pre-trained model fine-tuning using randomly generated multi-sensor IMU data.
    
    """

    def __init__(self, data_list, train_val_test = 'train', train_sequence_length = 260, train_val_ratio = 0.8,
                 random_seed = 42, window_len = 120, hop_len = 60, mix_train_val_babies = False,
                 augment_train_data = False, aug_p_noise = 0.0, aug_p_dropout = 0.1, aug_p_rotation = 0.3,
                 aug_p_chandropout = 0.3, aug_p_time_warping = 0.0, data_sampling_rate=1.0,
                 include_artificial_labels=False, normalize_functionals_sample_level=False,
                 normalize_functionals_dataset_level=True, functionals_include_mean=True,
                 functionals_include_var=True, functionals_include_skew=True, functionals_include_kurtosis=True,
                 functionals_include_min=True, functionals_include_max=True, functionals_include_zcr=True,
                 functionals_include_acf_mean=True, functionals_include_acf_var=True,
                 functionals_include_acf_skew=True, functionals_include_acf_kurtosis=True):
        super().__init__()
        
        if train_val_test == 'train' and augment_train_data:
            self.augment = augment_train_data
            self.aug_p_noise = aug_p_noise
            self.aug_p_dropout = aug_p_dropout
            self.aug_p_rotation = aug_p_rotation
            self.aug_p_chandropout = aug_p_chandropout
            self.aug_p_time_warping = aug_p_time_warping
            self.window_len = window_len
            self.hop_len = hop_len
        else:
            self.augment = False
        
        X = []
        data_masks = []
        if not mix_train_val_babies and train_val_test != 'test':
            # We split our training and validation data so that baby-specific data is not included in both sets.
            num_train_babies = int(np.round(train_val_ratio*len(data_list)))
            train_val_babies_permutation = np.random.RandomState(seed=random_seed*2).permutation(len(data_list))
            if train_val_test == 'train':
                data_list = [data_list[i] for i in train_val_babies_permutation[:num_train_babies]]
            else:
                data_list = [data_list[i] for i in train_val_babies_permutation[num_train_babies:]]
        
        # We go through the data sequences one at a time and we append them to their appropriate lists.
        for baby_data in data_list:
            data_in = baby_data['X']
            data_mask = baby_data['Mask']
            num_sequences = data_in.shape[0] // train_sequence_length
            leftover_sequence_len = data_in.shape[0] % train_sequence_length
            if not mix_train_val_babies or train_val_test == 'test':
                for i in range(num_sequences):
                    X.append(data_in[i*train_sequence_length:(i+1)*train_sequence_length,:,:])
                    data_masks.append(data_mask[i*train_sequence_length:(i+1)*train_sequence_length])
            else:
                num_train_seq = int(np.round(train_val_ratio*num_sequences)) # The number of training data sequences
                train_val_permutation = np.random.RandomState(seed=random_seed).permutation(num_sequences)
                if train_val_test == 'train':
                    sequences = train_val_permutation[:num_train_seq]
                else:
                    sequences = train_val_permutation[num_train_seq:]
                
                for i in sequences:
                    X.append(data_in[i*train_sequence_length:(i+1)*train_sequence_length,:,:])
                    data_masks.append(data_mask[i*train_sequence_length:(i+1)*train_sequence_length])
                
            if leftover_sequence_len != 0 and (train_val_test != 'validation' or not mix_train_val_babies):
                # We add the last sequence that is shorter than others and pad it to be of equal length
                X_leftover = np.copy(data_in[i*train_sequence_length:(i+1)*train_sequence_length,:,:])
                X_leftover[:leftover_sequence_len] = data_in[-leftover_sequence_len:, :, :]
                X.append(X_leftover)
                leftover_mask = np.ones_like(data_mask[i*train_sequence_length:(i+1)*train_sequence_length])
                leftover_mask[:leftover_sequence_len] = data_mask[-leftover_sequence_len:]
                data_masks.append(leftover_mask)
        
        self.X = np.array(X)
        self.data_masks = np.array(data_masks)
        
        # We compute functionals of the features
        feats_functionals = []
        for feat in X:
            functionals = []
            if functionals_include_mean:
                functionals.append(np.mean(feat, axis=2))
            if functionals_include_var:
                functionals.append(np.var(feat, axis=2))
            if functionals_include_skew:
                functionals.append(scipy.stats.skew(feat, axis=2))
            if functionals_include_kurtosis:
                functionals.append(scipy.stats.kurtosis(feat, axis=2))
            if functionals_include_min:
                functionals.append(feat.min(axis=2))
            if functionals_include_max:
                functionals.append(feat.max(axis=2))
            if functionals_include_zcr:
                functionals.append(librosa.zero_crossings(feat, axis=2).sum(axis=2) / window_len)
            if functionals_include_acf_mean or functionals_include_acf_var or functionals_include_acf_skew or functionals_include_acf_kurtosis:
                if functionals_include_acf_mean:
                    ac_channel_mean = []
                if functionals_include_acf_var:
                    ac_channel_var = []
                if functionals_include_acf_skew:
                    ac_channel_skew = []
                if functionals_include_acf_kurtosis:
                    ac_channel_kurtosis = []
                for i in range(feat.shape[1]):
                    feat_channel = feat[:, i, :]
                    ac = estimated_autocorrelation(feat_channel)
                    if functionals_include_acf_mean:
                        ac_channel_mean.append(np.mean(ac, axis=1))
                    if functionals_include_acf_var:
                        ac_channel_var.append(np.var(ac, axis=1))
                    if functionals_include_acf_skew:
                        ac_channel_skew.append(scipy.stats.skew(ac, axis=1))
                    if functionals_include_acf_kurtosis:
                        ac_channel_kurtosis.append(scipy.stats.kurtosis(ac, axis=1))
                if functionals_include_acf_mean:
                    ac_channel_mean = np.transpose(np.array(ac_channel_mean))
                    functionals.append(ac_channel_mean)
                if functionals_include_acf_var:
                    ac_channel_var = np.transpose(np.array(ac_channel_var))
                    functionals.append(ac_channel_var)
                if functionals_include_acf_skew:
                    ac_channel_skew = np.transpose(np.array(ac_channel_skew))
                    functionals.append(ac_channel_skew)
                if functionals_include_acf_kurtosis:
                    ac_channel_kurtosis = np.transpose(np.array(ac_channel_kurtosis))
                    functionals.append(ac_channel_kurtosis)
            functionals = np.stack(functionals, axis=2)
            
            # We reshape the functional array from the shape [train_sequence_length, num_channels, num_functionals]
            # into the shape [train_sequence_length, num_channels * num_functionals]
            functionals = functionals.reshape(functionals.shape[0], -1)
            
            if normalize_functionals_sample_level:
                feats_functionals.append(normalize_sample(functionals))
            else:
                feats_functionals.append(functionals)
        
        feats_functionals = np.array(feats_functionals)
        
        if normalize_functionals_dataset_level:
            feats_functionals = normalize_dataset(feats_functionals)
        
        self.feats_functionals = feats_functionals
        
        # We create artificial labels for our randomly generated dataset. There are nine different labels
        # for movement in MAIJU data.
        if include_artificial_labels:
            Y = np.zeros((len(self.X), train_sequence_length, 9))
            for i in range(len(Y)):
                for j in range(train_sequence_length):
                    random_vec = np.random.rand(Y.shape[2])
                    max_ind = np.argmax(random_vec)
                    Y[i, j, max_ind] = 1.0
        
            self.Y = Y
        
        self.include_artificial_labels = include_artificial_labels
        
        if data_sampling_rate < 1.00 and train_val_test != 'test':
            # We randomly select a subset of the data
            num_sampled = int(data_sampling_rate * len(X))
            np.random.seed(3*random_seed)
            sampling_indices = np.random.choice(np.arange(len(X)), num_sampled, replace=False)
            self.X = self.X[sampling_indices, :, :, :]
            self.data_masks = self.data_masks[sampling_indices, :]
            self.feats_functionals = self.feats_functionals[sampling_indices, :, :]
            if include_artificial_labels:
                self.Y = self.Y[sampling_indices, :, :]

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index):
        
        if self.augment:
            X = data_augmentation(self.X[index], self.aug_p_noise, self.aug_p_dropout, self.aug_p_rotation,
                                  self.aug_p_chandropout, self.aug_p_time_warping, self.window_len, self.hop_len)
        else:
            X = self.X[index]
        
        if self.include_artificial_labels:
            target_labels = self.Y[index]
        else:
            target_labels = 0
        
        return X, target_labels, self.data_masks[index], self.feats_functionals[index]




class sleep_edf_expanded_dataset_pfml(Dataset):
    """
    Dataloader for PFML pre-training using the pre-processed Sleep-EDF Database Expanded dataset
    (https://github.com/emadeldeen24/AttnSleep).
    
    """

    def __init__(self, data_dir = './sleep_edf_78', preprocess_data = False,
                 preprocessed_data_dir = './preprocessed_sleep_edf_exp_files_framed',
                 precompute_functionals = False, functionals_save_dir = './precomputed_sleep_edf_exp_functionals',
                 train_val_test = 'train', train_val_ratio = 0.8, random_seed = 42, fs=100,
                 window_len_seconds=4.0, hop_len_seconds=2.0, normalize_functionals_sample_level=False, 
                 normalize_functionals_dataset_level=True, functionals_include_mean=True,
                 functionals_include_var=True, functionals_include_skew=True, functionals_include_kurtosis=True,
                 functionals_include_min=True, functionals_include_max=True, functionals_include_zcr=True,
                 functionals_include_acf_mean=True, functionals_include_acf_var=True,
                 functionals_include_acf_skew=True, functionals_include_acf_kurtosis=True, data_sampling_rate=1.0):
        super().__init__()
        
        frames_path = os.path.join(preprocessed_data_dir, 'frames.npy')
        labels_path = os.path.join(preprocessed_data_dir, 'labels.npy')
        subject_ids_path = os.path.join(preprocessed_data_dir, 'subject_ids.npy')
        functionals_path = os.path.join(functionals_save_dir, 'functionals.npy')
        
        frame_len = int(window_len_seconds * fs)
        shift = int(hop_len_seconds * fs)
        
        # Preprocess the data
        if not os.path.exists(preprocessed_data_dir):
            os.makedirs(preprocessed_data_dir)
            preprocess_data = True
        else:
            if not all(os.path.exists(path) for path in [frames_path, labels_path, subject_ids_path]):
                preprocess_data = True
            if preprocess_data and len(os.listdir(preprocessed_data_dir)) != 0:
                # Remove old files from the given directory
                filenames_old_files = os.listdir(preprocessed_data_dir)
                for filename in filenames_old_files:
                    os.remove(os.path.join(preprocessed_data_dir, filename))
        
        if preprocess_data or len(os.listdir(preprocessed_data_dir)) == 0:
            # Find out our EDF files in the given directory
            try:
                filenames_edf = os.listdir(data_dir)
            except FileNotFoundError:
                sys.exit(f'Given EDF file directory {data_dir} does not exist!')
            
            # Remove other files that EDF files
            edf_file_names = [filename for filename in filenames_edf if filename.endswith('.npz')]
            edf_file_names = sorted(edf_file_names)
            del filenames_edf
            
            # We first determine the total number of sequences and the framed data shape
            num_sequences = 0
            num_frames = None
            num_channels = None
            
            for filename in edf_file_names:
                with np.load(os.path.join(data_dir, filename)) as loaded_data:
                    num_sequences += len(loaded_data['y'])
            
                    if num_frames is None:
                        X_example = loaded_data['x'].squeeze()
            
                        # If the file contains only one sequence, squeeze() removes the
                        # sequence dimension, so we add it back.
                        if X_example.ndim == 1:
                            X_example = np.expand_dims(X_example, axis=0)
            
                        if X_example.ndim < 3:
                            sequence_length = X_example.shape[1]
                            num_channels = 1
                        else:
                            sequence_length = X_example.shape[2]
                            num_channels = X_example.shape[1]
            
                        num_frames = int(np.floor(((sequence_length - frame_len) / shift) + 1))
                
            # Pre-allocate the framed sequences, labels and subject IDs
            frames = np.lib.format.open_memmap(frames_path, mode='w+', dtype=np.float32, shape=(num_sequences, num_frames, num_channels, frame_len))
            labels = np.lib.format.open_memmap(labels_path, mode='w+', dtype=np.float64, shape=(num_sequences,))
            subject_ids = np.lib.format.open_memmap(subject_ids_path, mode='w+', dtype='<U2', shape=(num_sequences,))
            
            # Go through each EDF file, preprocess the data, and write it into the arrays
            data_index = 0
            for filename in edf_file_names:
                with np.load(os.path.join(data_dir, filename)) as loaded_data:
                    X = loaded_data['x'].squeeze()
                    Y = loaded_data['y']
                
                # If the file contains only one sequence, squeeze() removes the sequence dimension, so we add it back.
                if X.ndim == 1:
                    X = np.expand_dims(X, axis=0)
                
                # X is now of shape [num_sequences, sequence_length]. We z-score normalize each sequence
                # to have zero mean and unit variance.
                for i in range(len(X)):
                    X[i,:] = (X[i,:] - X[i,:].mean()) / X[i,:].std()
                
                # We frame each sequence
                data_framed = frame_sig(X, frame_len, shift, sequence_batch=True)
                del X
                
                # Save the sequences
                for i in range(len(data_framed)):
                    frames[data_index] = data_framed[i,:,:,:]
                    labels[data_index] = Y[i]
                    subject_ids[data_index] = filename[3:5]
                    data_index += 1
            
            frames.flush()
            labels.flush()
            subject_ids.flush()
            del frames
            del labels
            del subject_ids
            
        # Load our preprocessed data using memory mapping
        self.preprocessed_data = np.load(frames_path, mmap_mode='r')
        
        # Split our data into separate sets
        np.random.seed(random_seed)
        mask_trainval_split = np.random.rand(len(self.preprocessed_data)) <= train_val_ratio
        
        # train_val_test has three options: 'train', 'validation' and 'test'. We use 'test' when we want to extract
        # features using a PFML pre-trained model, i.e. we use all of our data with the option 'test'.
        if train_val_test == 'train':
            self.feat_indices = np.arange(len(self.preprocessed_data))[mask_trainval_split]
        elif train_val_test == 'validation':
            self.feat_indices = np.arange(len(self.preprocessed_data))[~mask_trainval_split]
        else:
            self.feat_indices = np.arange(len(self.preprocessed_data))
        
        # Pre-compute the functionals
        if not os.path.exists(functionals_save_dir):
            os.makedirs(functionals_save_dir)
            precompute_functionals = True
        else:
            if not os.path.exists(functionals_path):
                precompute_functionals = True
            if precompute_functionals and len(os.listdir(functionals_save_dir)) != 0:
                # Remove old files from the given directory
                filenames_old_files = os.listdir(functionals_save_dir)
                for filename in filenames_old_files:
                    os.remove(os.path.join(functionals_save_dir, filename))
        
        if precompute_functionals or len(os.listdir(functionals_save_dir)) == 0:
            # We go through each frame one at a time and we compute its functionals
            feats_functionals = None
            for i in range(len(self.preprocessed_data)):
                feat = self.preprocessed_data[i].squeeze()
                functionals = []
                if functionals_include_mean:
                    functionals.append(np.mean(feat, axis=1))
                if functionals_include_var:
                    functionals.append(np.var(feat, axis=1))
                if functionals_include_skew:
                    functionals.append(scipy.stats.skew(feat, axis=1))
                if functionals_include_kurtosis:
                    functionals.append(scipy.stats.kurtosis(feat, axis=1))
                if functionals_include_min:
                    functionals.append(feat.min(axis=1))
                if functionals_include_max:
                    functionals.append(feat.max(axis=1))
                if functionals_include_zcr:
                    functionals.append(librosa.zero_crossings(feat, axis=1).sum(axis=1) / frame_len)
                if functionals_include_acf_mean or functionals_include_acf_var or functionals_include_acf_skew or functionals_include_acf_kurtosis:
                    ac = estimated_autocorrelation(feat)
                    if functionals_include_acf_mean:
                        functionals.append(np.mean(ac, axis=1))
                    if functionals_include_acf_var:
                        functionals.append(np.var(ac, axis=1))
                    if functionals_include_acf_skew:
                        functionals.append(scipy.stats.skew(ac, axis=1))
                    if functionals_include_acf_kurtosis:
                        functionals.append(scipy.stats.kurtosis(ac, axis=1))
                functionals = np.stack(functionals, axis=1)
                if normalize_functionals_sample_level:
                    functionals = normalize_sample(functionals)
                if feats_functionals is None:
                    feats_functionals = np.lib.format.open_memmap(functionals_path, mode='w+', dtype=functionals.dtype, shape=(len(self.preprocessed_data), functionals.shape[0], functionals.shape[1]))
                feats_functionals[i] = functionals
            
            if normalize_functionals_dataset_level:
                feats_functionals = normalize_dataset(feats_functionals)
            feats_functionals.flush()
            del feats_functionals
            
        self.preprocessed_functionals = np.load(functionals_path, mmap_mode='r')
        self.functional_indices = self.feat_indices.copy()
        
        if data_sampling_rate < 1.00 and train_val_test != 'test':
            # We randomly select a subset of the data
            num_sampled = int(data_sampling_rate * len(self.feat_indices))
            np.random.seed(3*random_seed)
            sampling_indices = np.random.choice(np.arange(len(self.feat_indices)), num_sampled, replace=False)
            self.feat_indices = self.feat_indices[sampling_indices]
            self.functional_indices = self.functional_indices[sampling_indices]

    def __len__(self) -> int:
        return len(self.feat_indices)
    
    def __getitem__(self, index):
        
        X = np.array(self.preprocessed_data[self.feat_indices[index]], copy=True)
        feats_functionals = np.array(self.preprocessed_functionals[self.functional_indices[index]], copy=True)
        data_mask = np.zeros((len(X)))
        
        return X, data_mask, feats_functionals







class random_speech_data_dataset(Dataset):
    """
    Dataloader for PFML pre-training and pre-trained model fine-tuning using randomly generated speech data.
    
    """

    def __init__(self, data_list, train_val_test = 'train', train_val_ratio = 0.8, random_seed = 42, 
                 data_sampling_rate=1.0, normalize_waveform=True, window_len_seconds=0.03, hop_len_seconds=0.01,
                 fs=16000, max_length_seconds=3.0, include_artificial_labels=True):
        super().__init__()
        
        
        X = []
        max_num_samples = int(max_length_seconds * fs)
        frame_len = int(window_len_seconds * fs)
        shift = int(hop_len_seconds * fs)
        
        for x in data_list:
            
            # Normalize to zero mean, unit variance
            if normalize_waveform:
                x = (x - x.mean()) / x.std()
            
            # We either truncate or zero-pad our signal to be of the length max_length_seconds
            if len(x) != max_num_samples:
                x = librosa.util.fix_length(x, size=max_num_samples)
            
            # We frame our signal. x_framed is of size [num_frames, frame_len]
            x_framed = librosa.util.frame(x, frame_length=frame_len, hop_length=shift, axis=0)
            
            X.append(x_framed)
        
        self.X = np.array(X)
        
        # We create artificial binary labels for our randomly generated dataset.
        if include_artificial_labels:
            Y = np.zeros((len(self.X), 2))
            for i in range(len(Y)):
                random_vec = np.random.rand(Y.shape[1])
                max_ind = np.argmax(random_vec)
                Y[i, max_ind] = 1.0
            self.Y = Y
        
        self.include_artificial_labels = include_artificial_labels
        
        if train_val_test != 'test':
            mask_trainval_split = np.random.rand(len(self.X)) <= train_val_ratio
            if train_val_test == 'train':
                self.X = self.X[mask_trainval_split]
                if include_artificial_labels:
                    self.Y = self.Y[mask_trainval_split]
            else:
                self.X = self.X[~mask_trainval_split]
                if include_artificial_labels:
                    self.Y = self.Y[~mask_trainval_split]
        
        if data_sampling_rate < 1.00 and train_val_test != 'test':
            # We randomly select a subset of the data
            num_sampled = int(data_sampling_rate * len(X))
            np.random.seed(3*random_seed)
            sampling_indices = np.random.choice(np.arange(len(X)), num_sampled, replace=False)
            self.X = self.X[sampling_indices, :, :]
            if include_artificial_labels:
                self.Y = self.Y[sampling_indices, :]

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index):
        
        if self.include_artificial_labels:
            return self.X[index], self.Y[index]
        else:
            return self.X[index], 0










class sleep_edf_expanded_dataset_pfml_finetuning(Dataset):
    """
    Dataloader for fine-tuning PFML pre-trained models using the pre-processed Sleep-EDF Database Expanded dataset
    (https://github.com/emadeldeen24/AttnSleep).
    
    """
    
    def __init__(self, test_subject_index_list, preprocessed_data_dir = './preprocessed_sleep_edf_exp_files_framed',
                 train_val_test = 'train', train_val_ratio = 0.8, random_seed = 42, mix_train_val_subjects = False,
                 data_sampling_rate=1.0):
        super().__init__()
        
        # Load the framed sequences (using Numpy's memory mapping), labels, and subject IDs
        self.X = np.load(os.path.join(preprocessed_data_dir, 'frames.npy'), mmap_mode='r')
        self.Y = np.load(os.path.join(preprocessed_data_dir, 'labels.npy'))
        subject_ids = np.load(os.path.join(preprocessed_data_dir, 'subject_ids.npy'))
        
        if not mix_train_val_subjects and train_val_test != 'test':
            # We split our training and validation data so that test subject-specific data is not included in both sets.
            num_train_test_subjects = int(np.round(train_val_ratio*len(test_subject_index_list))) # The number of training data sequences
            train_val_test_subjects_permutation = np.random.RandomState(seed=random_seed*2).permutation(len(test_subject_index_list))
            if train_val_test == 'train':
                test_subject_index_list = [test_subject_index_list[i] for i in train_val_test_subjects_permutation[:num_train_test_subjects]]
            else:
                test_subject_index_list = [test_subject_index_list[i] for i in train_val_test_subjects_permutation[num_train_test_subjects:]]
        
        # Find the data sequences belonging to the selected subjects
        selected_data_indices = np.where(np.isin(subject_ids, np.array(test_subject_index_list, dtype=str)))[0]
        
        if not mix_train_val_subjects or train_val_test == 'test':
            self.indices = selected_data_indices
        else:
            np.random.seed(random_seed*5)
            mask_trainval_split = np.random.rand(len(selected_data_indices)) <= train_val_ratio
            if train_val_test == 'train':
                self.indices = selected_data_indices[mask_trainval_split]
            else:
                self.indices = selected_data_indices[~mask_trainval_split]
        
        if data_sampling_rate < 1.00 and train_val_test != 'test':
            # We randomly select a subset of the data
            num_sampled = int(data_sampling_rate * len(self.indices))
            np.random.seed(3*random_seed)
            sampling_indices = np.random.choice(np.arange(len(self.indices)), num_sampled, replace=False)
            self.indices = self.indices[sampling_indices]
    
    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index):
        
        data_index = self.indices[index]
        X = np.array(self.X[data_index], copy=True)
        data_mask = np.zeros((len(X)))
        
        return X, self.Y[data_index], data_mask





def normalize_sample(feats) -> np.ndarray:
    
    normalized = (feats - feats.mean(axis=0)) / feats.std(axis=0)
    
    # Remove NaN values by converting them to zero
    normalized = np.nan_to_num(normalized)
    
    return normalized


# Normalize the feature array to have zero mean and unit variance along each feature. The dimensions
# of feats are either (sample_index, frame_index, feature_index) or (frame_index, feature_index), the
# latter of which is for variable-length samples stored consecutively.
def normalize_dataset(feats, sample_offsets=None):

    if sample_offsets is None:
        feats_unrolled = feats.reshape(-1, feats.shape[-1])
    else:
        feats_unrolled = feats

    # We compute the statistics one feature at a time so that the complete feature array
    # does not have to be copied into RAM.
    feat_mean = np.empty(feats_unrolled.shape[-1], dtype=feats_unrolled.dtype)
    feat_std = np.empty(feats_unrolled.shape[-1], dtype=feats_unrolled.dtype)

    for i in range(feats_unrolled.shape[-1]):
        feat = np.nan_to_num(feats_unrolled[:, i])
        feat_mean[i] = feat.mean()
        feat_std[i] = feat.std()

    if sample_offsets is None:
        # Fixed-length data
        for i in range(len(feats)):
            feats[i] = (feats[i] - feat_mean) / feat_std
            feats[i] = np.nan_to_num(feats[i])
    else:
        # Variable-length samples stored consecutively
        for i in range(len(sample_offsets) - 1):
            start = sample_offsets[i]
            stop = sample_offsets[i + 1]
            feats[start:stop] = (feats[start:stop] - feat_mean) / feat_std
            feats[start:stop] = np.nan_to_num(feats[start:stop])

    return feats




def estimated_autocorrelation(frames):
    
    ac = []
    for x in frames:
        n = len(x)
        variance = x.var()
        x = x - x.mean()
        r = np.correlate(x, x, mode = 'full')[-n:]
        result = r/(variance*(np.arange(n, 0, -1)))
        ac.append(result)
        
    return np.array(ac)



def time_warping(data, p=1.0, winlen=120):
    basevec = np.arange(winlen) + 1.0
    num_frames = int(np.floor(((data.shape[0] - winlen)/winlen) + 1))
    for iFrame in range(num_frames):
        # Randomly warp p*100% of frames
        if np.random.random_sample() <= p:
        # Random sinusoid with random phase, amplitude [0.5, 1.5], frequency
            freq = np.random.random_sample() * basevec / basevec.shape[0]
            phase = 2 * np.pi * np.random.random_sample()
            amplitude = np.random.random_sample()
            sinusoid = amplitude * np.sin(2 * np.pi * freq + phase) + 2
            sinusoid /= np.mean(sinusoid)

            newbase = np.cumsum(sinusoid)
            start = iFrame * winlen
            stop = start + winlen
            for iChan in range(data.shape[1]):
                data[start:stop,iChan] = np.interp(newbase, basevec, data[start:stop, iChan])

    return data



def rotationMatrix(a_x, a_y, a_z, angle_type='deg'):
    if angle_type == 'deg':
        a_x *= np.pi / 180.0
        a_y *= np.pi / 180.0
        a_z *= np.pi / 180.0

    M = np.array([[np.cos(a_y) * np.cos(a_z), 
                   -np.cos(a_x) * np.sin(a_z) + np.sin(a_x) * np.sin(a_y) * np.cos(a_z), 
                   np.sin(a_x) * np.sin(a_z) + np.cos(a_x) * np.sin(a_y) * np.cos(a_z)],
                  [np.cos(a_y) * np.sin(a_z), 
                   np.cos(a_x) * np.cos(a_z) + np.sin(a_x) * np.sin(a_y) * np.sin(a_z),
                   -np.sin(a_x) * np.cos(a_z) + np.cos(a_x) * np.sin(a_y) * np.sin(a_z)],
                  [-np.sin(a_y),
                   np.sin(a_x) * np.cos(a_y),
                   np.cos(a_x) * np.cos(a_y)]])

    return M




def random_rotation(data, angle=15.0):
    # Get rotation matrix, random rotation for each sensor
    range_x = [-angle, angle]
    range_y = [-angle, angle]
    range_z = [-angle, angle]
    Nsens = data.shape[1] // 6
    n = data.shape[-1] // 2
    acc = data[:,:n]
    gyro = data[:,n:]
    for i in range(Nsens):
        a_x = np.random.random_sample() * (range_x[1] - range_x[0]) + range_x[0]
        a_y = np.random.random_sample() * (range_y[1] - range_y[0]) + range_y[0]
        a_z = np.random.random_sample() * (range_z[1] - range_z[0]) + range_z[0]
        M = rotationMatrix(a_x, a_y, a_z)
        
        acc[:,i*3:(i+1)*3] = np.matmul(acc[:,i*3:(i+1)*3], M)
        gyro[:,i*3:(i+1)*3] = np.matmul(gyro[:,i*3:(i+1)*3], M)

    data = np.concatenate([acc, gyro], axis=-1)

    return data




def dropout_noise(data, p):
    mask = np.random.binomial(1, 1.0 - p, data.shape)
    
    return data * mask



def channel_dropout(data, num_chans=1, tot_chans=4):
    chans_to_drop = np.random.permutation(tot_chans)
    chans_to_drop = chans_to_drop[:num_chans]
    N = data.shape[-1] // 2
    for i in chans_to_drop:
        data[:,(3*i):(3*i+3)] *= 0.0 # Accelerometer signals
        data[:,(N+3*i):(N+3*i+3)] *= 0.0 # Gyroscope signals

    return data



def frame_sig(X, winlen, hop, sequence_batch=False):
    """
    Frame either one multi-channel signal or a batch of sequences.
    
    If sequence_batch is False (default), X is interpreted as one signal with dimensions 
    [sequence_length, num_channels] and the output is of shape [num_frames, num_channels, winlen].
    
    If sequence_batch is True, X is interpreted as a batch of sequences with dimensions
    [num_sequences, sequence_length] or [num_sequences, num_channels, sequence_length]
    and the output is of shape [num_sequences, num_frames, num_channels, winlen].

    """

    if sequence_batch:
        if X.ndim == 2:
            # Input shape = [num_sequences, sequence_length]
            X = np.expand_dims(X, axis=1)

        elif X.ndim != 3:
            sys.exit('When sequence_batch=True, X must have shape [num_sequences, sequence_length] or [num_sequences, num_channels, sequence_length].')

        num_sequences = X.shape[0]
        num_channels = X.shape[1]
        sequence_length = X.shape[2]

        num_frames = int(np.floor(((sequence_length - winlen) / hop) + 1))
        X_framed = np.zeros((num_sequences, num_frames, num_channels, winlen), dtype=np.float32)

        for sequence_index in range(num_sequences):
            for frame_index in range(num_frames):
                start = frame_index * hop
                stop = start + winlen
                X_framed[sequence_index, frame_index, :, :] = X[sequence_index, :, start:stop]

    else:
        if X.ndim != 2:
            sys.exit('When sequence_batch=False, X must have shape [sequence_length, num_channels].')

        sequence_length = X.shape[0]
        num_channels = X.shape[1]

        num_frames = int(np.floor(((sequence_length - winlen) / hop) + 1))
        X_framed = np.zeros((num_frames, num_channels, winlen), dtype=np.float32)

        for frame_index in range(num_frames):
            start = frame_index * hop
            stop = start + winlen
            X_framed[frame_index, :, :] = np.transpose(X[start:stop, :])

    return X_framed






def data_augmentation(data, aug_p_noise, aug_p_dropout, aug_p_rotation, aug_p_chandropout,
                      aug_p_time_warping, window_len, hop_len):
    
    # Augmentation to frames, assume data is 50% overlapped
    N = data.shape[-1] // 2
    data = np.concatenate([np.reshape(np.transpose(data[:,:,:N], [0,2,1]), [-1, data.shape[1]]),
                           np.transpose(data[-1,:,N:])], axis=0)

    # Time warping
    if np.random.random_sample() < aug_p_time_warping:
        data = time_warping(data, p=1.0, winlen=window_len)

    # Random rotation
    if np.random.random_sample() < aug_p_rotation:
        data = random_rotation(data)

    # Additive noise augmentation
    if np.random.random_sample() < aug_p_noise:
        data = dropout_noise(data, aug_p_dropout)

    # Sensor dropout
    if np.random.random_sample() < aug_p_chandropout:
        data = channel_dropout(data, num_chans=1)

    # Retain framed format
    data = frame_sig(data, window_len, hop_len)

    return data


