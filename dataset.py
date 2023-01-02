import sys
import utils
import librosa
import random
import glob
import os

import numpy as np
import tensorflow as tf

from tqdm import tqdm

SEED = 2022
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'


def file_load(file_name):
    try:
        return librosa.load(file_name, sr=None, mono=False)
    except:
        print("file_broken or not exists!! : {}".format(file_name))


def file_to_log_mel_spectrogram(args, file_name):
    y, sr = file_load(file_name)
    y = y[:args.sr * 10]
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=args.n_fft, hop_length=args.hop_length,
                                                     n_mels=args.n_mels, power=args.power)
    log_mel_spectrogram = 20.0 / args.power * np.log10(mel_spectrogram + sys.float_info.epsilon)
    return log_mel_spectrogram


def list_to_dataset(args, file_list):
    label = []
    num_data = len(file_list)
    dataset_array = np.zeros((num_data, args.n_mels, args.frames))
    for ii in tqdm(range(num_data), desc='generate dataset'):
        log_mel = file_to_log_mel_spectrogram(args, file_list[ii])
        id_num = utils.get_id_num(file_list[ii])
        label.append(id_num)
        dataset_array[ii] = log_mel
    label = np.array(label)
    return dataset_array, label


def generate_train_dataset(args, path):
    train_dataset_path = f'{args.pre_data_dir}/train_data_{os.path.split(path)[1]}.npy'
    train_dataset_label_path = f'{args.pre_data_dir}/train_data_{os.path.split(path)[1]}_label.npy'
    if not os.path.isfile(train_dataset_path) or not os.path.isfile(train_dataset_label_path):
        train_data_path = f'{path}/train/*.wav'
        file_list = utils.file_list_generator(train_data_path)
        data, label = list_to_dataset(args, file_list)
        np.save(train_dataset_path, data)
        np.save(train_dataset_label_path, label)


def generate_eval_dataset(args, path):
    machine_type = os.path.split(path)[1]
    id_list = utils.get_machine_id_list(os.path.join(path, 'test'))

    for id_str in id_list:
        id_path = f'{args.pre_data_dir}/eval_data_{machine_type}_{id_str}.npy'
        id_label_path = f'{args.pre_data_dir}/eval_data_{machine_type}_{id_str}_label.npy'
        if not os.path.isfile(id_path) or not os.path.isfile(id_label_path):
            normal_list = sorted(glob.glob(f'{args.dataset_dir}/{machine_type}/test/normal_{id_str}*.wav'))
            anomaly_list = sorted(glob.glob(f'{args.dataset_dir}/{machine_type}/test/anomaly_{id_str}*.wav'))
            file_list = np.concatenate((normal_list, anomaly_list), axis=0)
            data, label = list_to_dataset(args, file_list)
            np.save(id_path, data)
            np.save(id_label_path, label)


def generate_test_dataset(args, path):
    machine_type = os.path.split(path)[1]
    id_list = utils.get_machine_id_list(os.path.join(args.test_dir, os.path.split(path)[1], 'test'))
    for id_str in id_list:
        id_path = f'{args.pre_data_dir}/test_data_{machine_type}_{id_str}.npy'
        id_label_path = f'{args.pre_data_dir}/test_data_{machine_type}_{id_str}_label.npy'
        if not os.path.isfile(id_path) or not os.path.isfile(id_label_path):
            normal_list = sorted(glob.glob(f'{args.test_dir}/{machine_type}/test/normal_{id_str}*.wav'))
            anomaly_list = sorted(glob.glob(f'{args.test_dir}/{machine_type}/test/anomaly_{id_str}*.wav'))
            file_list = np.concatenate((normal_list, anomaly_list), axis=0)
            data, label = list_to_dataset(args, file_list)
            np.save(id_path, data)
            np.save(id_label_path, label)


def load_dataset(args, path, dataset='train'):
    dataset_path = f'{args.pre_data_dir}/{dataset}_data_{os.path.split(path)[1]}.npy'
    dataset_label_path = f'{args.pre_data_dir}/{dataset}_data_{os.path.split(path)[1]}_label.npy'
    data = np.load(dataset_path)
    label = np.load(dataset_label_path)
    return data, label


def load_eval_dataset(args, machine_type, id_str):
    dataset_path = f'{args.pre_data_dir}/eval_data_{machine_type}_{id_str}.npy'
    dataset_label_path = f'{args.pre_data_dir}/eval_data_{machine_type}_{id_str}_label.npy'
    data = np.load(dataset_path)
    label = np.load(dataset_label_path)
    return data, label


def load_eval_dataset_name(args, machine_type, id_str):
    normal_files = sorted(glob.glob(f'{args.dataset_dir}/{machine_type}/test/normal_{id_str}*.wav'))
    anomaly_files = sorted(glob.glob(f'{args.dataset_dir}/{machine_type}/test/anomaly_{id_str}*.wav'))
    names = np.concatenate((normal_files, anomaly_files), axis=0)
    return names


def load_test_dataset(args, machine_type, id_str):
    dataset_path = f'{args.pre_data_dir}/test_data_{machine_type}_{id_str}.npy'
    dataset_label_path = f'{args.pre_data_dir}/test_data_{machine_type}_{id_str}_label.npy'
    data = np.load(dataset_path)
    label = np.load(dataset_label_path)
    return data, label


def load_test_dataset_name(args, machine_type, id_str):
    normal_files = sorted(glob.glob(f'{args.test_dir}/{machine_type}/test/normal_{id_str}*.wav'))
    anomaly_files = sorted(glob.glob(f'{args.test_dir}/{machine_type}/test/anomaly_{id_str}*.wav'))
    names = np.concatenate((normal_files, anomaly_files), axis=0)
    return names


def dataset_frequency_normalizing(args, path, data):
    mean_path = f'{args.pre_data_dir}/train_dataset_{os.path.split(path)[1]}_mean.npy'
    std_path = f'{args.pre_data_dir}/train_dataset_{os.path.split(path)[1]}_std.npy'

    if os.path.isfile(mean_path):
        mean = np.load(mean_path)
    else:
        mean = np.mean(data, axis=(0, 2))
        np.save(mean_path, mean)
    if os.path.isfile(std_path):
        std = np.load(std_path)
    else:
        std = np.std(data, axis=(0, 2))
        np.save(std_path, std)
    for freq in range(data.shape[1]):
        data[:, freq, :] = (data[:, freq, :] - mean[freq]) / std[freq]
    return data


def label_modification(label, n_class=None, machine_type=None):
    label_list = []
    [label_list.append(v) for v in label if v not in label_list]
    label_list.sort()
    if n_class is None:
        output_label = np.array([label_list.index(label[i]) for i, v in enumerate(label)])
        train_label = np.zeros((output_label.size, output_label.max() + 1))
        train_label[np.arange(output_label.size), output_label] = 1
    else:
        if machine_type == 'ToyCar' or machine_type == 'ToyConveyor':
            label -= 1
        train_label = np.zeros((label.size, n_class))
        train_label[np.arange(label.size), label] = 1
    return train_label


def seg_label_modification(args, label, n_class=None, machine_type=None):
    label_list = []
    [label_list.append(v) for v in label if v not in label_list]
    label_list.sort()
    if n_class is None:
        output_label = np.array([label_list.index(label[i]) for i, v in enumerate(label)])
        train_label = np.zeros((output_label.size, args.frames, np.max(output_label) + 1))
        train_label[np.arange(output_label.size), :, output_label] = 1
    else:
        if machine_type == 'ToyCar' or machine_type == 'ToyConveyor':
            label -= 1
        train_label = np.zeros((label.size, args.frames, n_class))
        train_label[np.arange(label.size), :, label] = 1

    return train_label


def mixup(data, label):
    output_data, output_label = [], []
    num_data = data.shape[0]

    for ii in tqdm(range(num_data), desc='Aug 1 - label mixing'):
        jj = random.randint(0, num_data - 1)
        w = random.uniform(0, 1)
        dd = w * data[ii] + (1 - w) * data[jj]
        ll = w * label[ii] + (1 - w) * label[jj]
        output_data.append(dd)
        output_label.append(ll)
    return output_data, output_label


def id_segmentation(dataset, label):
    output_dataset, output_label = [], []
    num_data = dataset.shape[0]
    len_t = dataset.shape[1]
    for ii in tqdm(range(num_data), desc='Aug 2 - id segmentation'):
        seg_length = random.randint(0, int(len_t / 2) - 1)
        pri_idx = random.randint(0, len_t - seg_length - 1)
        sec_idx = random.randint(0, len_t - seg_length - 1)
        w = random.random()
        d = random.randint(0, num_data - 1)

        data = np.copy(dataset[ii])
        data_label = np.copy(label[ii])
        seg_data = np.copy(dataset[d])
        data_label1 = np.copy(label[d])

        # segmentation + label mixing
        data[pri_idx: pri_idx + seg_length] = w * seg_data[sec_idx: sec_idx + seg_length] + (1 - w) * data[pri_idx: pri_idx + seg_length]
        data_label[pri_idx: pri_idx + seg_length] = w * data_label1[sec_idx: sec_idx + seg_length] + (1 - w) * data_label[pri_idx: pri_idx + seg_length]

        output_dataset.append(data)
        output_label.append(data_label)

    return output_dataset, output_label


def dataset_augmentation(args, data, label):
    total_dataset, total_label = [], []

    if args.aug_orig:
        total_dataset.extend(list(data))
        total_label.extend(list(label))

    if args.aug_mixup:
        aug_data, aug_label = mixup(data, label)
        total_dataset = total_dataset + aug_data
        total_label = total_label + aug_label

    if args.aug_seg:
        aug_data, aug_label = id_segmentation(data, label)
        total_dataset = total_dataset + aug_data
        total_label = total_label + aug_label

    total_dataset = np.array(total_dataset)
    total_label = np.array(total_label)

    return total_dataset, total_label


def get_wavenet_eval_test_dataset(args, machine_type, id_str, is_eval=True):
    if is_eval:
        name = load_eval_dataset_name(args, machine_type, id_str)
        data, _ = load_eval_dataset(args, machine_type, id_str)
        y_true = utils.eval_label_generator(args, machine_type, id_str)
    else:
        name = load_test_dataset_name(args, machine_type, id_str)
        data, _ = load_test_dataset(args, machine_type, id_str)
        y_true = utils.test_label_generator(args, machine_type, id_str)

    data = dataset_frequency_normalizing(args, os.path.join(f'{args.pre_data_dir}', machine_type), data)
    batch_dataset = tf.data.Dataset.from_tensor_slices((data, name)).batch(args.batch_size)
    return batch_dataset, y_true



def load_wavenet_dataset(args, path):
    data, label = load_dataset(args, path)
    data = dataset_frequency_normalizing(args, path, data)
    data = np.moveaxis(data, 1, 2)
    label = label_modification(label)
    train_data, train_label = dataset_augmentation(args, data, label)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label)).shuffle(len(train_data)).batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((data, label)).batch(args.batch_size)

    return train_dataset, val_dataset



def get_resnet_eval_test_dataset(args, machine_type, id_str, n_class, is_eval=True):
    if is_eval:
        name = load_eval_dataset_name(args, machine_type, id_str)
        data, label = load_eval_dataset(args, machine_type, id_str)
        y_true = utils.eval_label_generator(args, machine_type, id_str)
    else:
        name = load_test_dataset_name(args, machine_type, id_str)
        data, label = load_test_dataset(args, machine_type, id_str)
        y_true = utils.test_label_generator(args, machine_type, id_str)

    data = dataset_frequency_normalizing(args, os.path.join(f'{args.pre_data_dir}', machine_type), data)
    label = label_modification(label, n_class, machine_type)
    data = data[..., np.newaxis]

    batch_dataset = tf.data.Dataset.from_tensor_slices((data, label, name)).batch(args.batch_size)
    return batch_dataset, y_true


def load_resnet_dataset(args, path):
    data, label = load_dataset(args, path)
    data = dataset_frequency_normalizing(args, path, data)
    label = label_modification(label)
    train_data, train_label = dataset_augmentation(args, data, label)
    train_data = train_data[..., np.newaxis]
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label)).shuffle(len(train_data)).batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((data, label)).batch(args.batch_size)
    return train_dataset, val_dataset


def get_mtl_class_eval_test_dataset(args, machine_type, id_str, n_class, is_eval=True, is_seg=False):
    if is_eval:
        name = load_eval_dataset_name(args, machine_type, id_str)
        data, label = load_eval_dataset(args, machine_type, id_str)
        y_true = utils.eval_label_generator(args, machine_type, id_str)
    else:
        name = load_test_dataset_name(args, machine_type, id_str)
        data, label = load_test_dataset(args, machine_type, id_str)
        y_true = utils.test_label_generator(args, machine_type, id_str)

    if is_seg:
        label = seg_label_modification(args, label, n_class, machine_type)
    else:
        label = label_modification(label, n_class, machine_type)

    data = dataset_frequency_normalizing(args, os.path.join(f'{args.pre_data_dir}', machine_type), data)
    batch_dataset = tf.data.Dataset.from_tensor_slices((data, label, name)).batch(args.batch_size)
    return batch_dataset, y_true


def load_mtl_class_dataset(args, path, is_seg=False):
    data, label = load_dataset(args, path)
    data = dataset_frequency_normalizing(args, path, data)
    data = np.moveaxis(data, 1, 2)
    if is_seg:
        label = seg_label_modification(args, label)
    else:
        label = label_modification(label)
    train_data, train_label = dataset_augmentation(args, data, label)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label)).shuffle(len(train_data)).batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((data, label)).batch(args.batch_size)
    return train_dataset, val_dataset


def get_mtl_class_seg_eval_test_dataset(args, machine_type, id_str, n_class, is_eval=True):
    if is_eval:
        name = load_eval_dataset_name(args, machine_type, id_str)
        data, label = load_eval_dataset(args, machine_type, id_str)
        y_true = utils.eval_label_generator(args, machine_type, id_str)
    else:
        name = load_test_dataset_name(args, machine_type, id_str)
        data, label = load_test_dataset(args, machine_type, id_str)
        y_true = utils.test_label_generator(args, machine_type, id_str)

    label1 = label_modification(label, n_class, machine_type)
    label2 = seg_label_modification(args, label, n_class, machine_type)

    data = dataset_frequency_normalizing(args, os.path.join(f'{args.pre_data_dir}', machine_type), data)
    batch_dataset = tf.data.Dataset.from_tensor_slices((data, label1, label2, name)).batch(args.batch_size)
    return batch_dataset, y_true


def load_mtl_class_seg_dataset(args, path):
    data, label = load_dataset(args, path)
    data = dataset_frequency_normalizing(args, path, data)
    data = np.moveaxis(data, 1, 2)
    label1 = label_modification(label)
    label2 = seg_label_modification(args, label)

    train_dataset = tf.data.Dataset.from_tensor_slices((data, label1, label2)).shuffle(len(data)).batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((data, label1, label2)).batch(args.batch_size)
    return train_dataset, val_dataset