import os
import glob
import itertools
import re
import csv
import numpy as np


def get_id_num(file_name):
    for id_num in range(10):
        string = 'id_0' + str(id_num)
        if string in file_name:
            return id_num
        else:
            pass
    return -1


def select_dirs(args):
    dir_path = os.path.abspath(f'{args.dataset_dir}/*')
    dirs = sorted(glob.glob(dir_path))
    output_dirs = [v for v in dirs if os.path.split(v)[1] in args.machines]
    return output_dirs


def file_list_generator(target_dir):
    files = sorted(glob.glob(target_dir))
    if len(files) == 0:
        print('no_wav_file!!')
    print('train_file num : {num}'.format(num=len(files)))
    return files


def get_machine_id_list(path):
    dir_path = os.path.abspath(f'{path}/*.wav')
    files_path = sorted(glob.glob(dir_path))
    machine_id_list = sorted(list(set(itertools.chain.from_iterable([re.findall('id_[0-9][0-9]', ext_id) for ext_id in files_path]))))
    return machine_id_list


def eval_label_generator(args, machine_type, id_str):
    normal_files = sorted(glob.glob(f'{args.dataset_dir}/{machine_type}/test/normal_{id_str}*.wav'))
    normal_labels = np.zeros(len(normal_files))
    anomaly_files = sorted(glob.glob(f'{args.dataset_dir}/{machine_type}/test/anomaly_{id_str}*.wav'))
    anomaly_labels = np.ones(len(anomaly_files))
    labels = np.concatenate((normal_labels, anomaly_labels), axis=0)
    return labels


def test_label_generator(args, machine_type, id_str):
    normal_files = sorted(glob.glob(f'{args.test_dir}/{machine_type}/test/normal_{id_str}*.wav'))
    normal_labels = np.zeros(len(normal_files))
    anomaly_files = sorted(glob.glob(f'{args.test_dir}/{machine_type}/test/anomaly_{id_str}*.wav'))
    anomaly_labels = np.ones(len(anomaly_files))
    labels = np.concatenate((normal_labels, anomaly_labels), axis=0)
    return labels


def save_model(args, model, machine_type, path, visualizer):
    save_model_path = f'{args.model_dir}/{args.version}/{machine_type}/{path}'
    model.save(save_model_path)
    history_img = f'{save_model_path}/history.png'
    visualizer.save_figure(history_img)


def save_csv(save_file_path,
             save_data):
    with open(save_file_path, 'w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)