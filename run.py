import argparse
import yaml

from visualizer import *
from model import *
from trainer import *

param_path = './param.yaml'
with open(param_path) as f:
    param = yaml.safe_load(f)


parser = argparse.ArgumentParser()

# path dir
parser.add_argument('--dataset-dir', default=param['dataset_dir'], type=str, help='dataset dir')
parser.add_argument('--test-dir', default=param['test_dir'], type=str, help='evaluation dataset dir')
parser.add_argument('--pre-data-dir', default=param['pre_data_dir'], type=str, help='preprocess data dir')
parser.add_argument('--model-dir', default=param['model_dir'], type=str, help='model dir')
parser.add_argument('--result-dir', default=param['result_dir'], type=str, help='result dir')
parser.add_argument('--result-file', default=param['result_file'], type=str, help='result file name')
parser.add_argument('--machines', default=param['machines'], nargs='+', type=str, help='allowed processing machine')

parser.add_argument('--seed', default=param['seed'], type=int, help='random seed')
# model dir
parser.add_argument('--training-mode', default=param['training_mode'], type=str)
parser.add_argument('--version', default=param['version'], type=str, help='version')
# spectrogram features
parser.add_argument('--sr', default=param['sr'], type=int, help='STFT sampling rate')
parser.add_argument('--n-fft', default=param['n_fft'], type=int, help='STFT n_fft')
parser.add_argument('--win-length', default=param['win_length'], type=int, help='STFT win length')
parser.add_argument('--hop-length', default=param['hop_length'], type=int, help='STFT hop length')
parser.add_argument('--n-mels', default=param['n_mels'], type=int, help='STFT n_mels')
parser.add_argument('--frames', default=param['frames'], type=int, help='STFT time frames')
parser.add_argument('--power', default=param['power'], type=float, help='STFT power')
# training
parser.add_argument('--batch-size', default=param['batch_size'], type=int, help='batch size')
parser.add_argument('--epochs', default=param['epochs'], type=int, help='training epochs')
parser.add_argument('--early-stop', default=param['early_stop'], type=int, help='number of epochs for early stopping')
parser.add_argument('--lr', default=param['lr'], type=float, help='initial learning rate')
parser.add_argument('--device-ids', default=param['device_ids'], nargs='+', type=int, help='gpu ids')
# model parameters
parser.add_argument('--channel-mul', default=param['channel_mul'], type=int, help='number of channel multiply')
parser.add_argument('--n-blocks', default=param['n_blocks'], type=int, help='number of residual blocks')
parser.add_argument('--n-groups', default=param['n_groups'], type=int, help='number of groups in conv layer')
parser.add_argument('--kernel-size', default=param['kernel_size'], type=int, help='conv kernel size')
# data augmentation
parser.add_argument('--aug-orig', default=param['aug_orig'], type=int, help='append original data')
parser.add_argument('--aug-mixup', default=param['aug_mixup'], type=int, help='append mixup data')
parser.add_argument('--aug-seg', default=param['aug_seg'], type=int, help='append seg data')

def set_random_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(seed)
    np.random.seed(seed)


def preprocess():
    dirs = utils.select_dirs(args)
    for path in dirs:
        dataset.generate_train_dataset(args, path)
        dataset.generate_eval_dataset(args, path)
        dataset.generate_test_dataset(args, path)


def train_wavenet():
    dirs = utils.select_dirs(args)
    mean_list, inv_cov_list = [], []
    for idx, target_dir in enumerate(dirs):
        print('\n' + '=' * 60)
        print(f'[{idx + 1}/{len(dirs)}] {target_dir}')

        machine_type = os.path.split(target_dir)[1]

        visualizer = WaveNetVisualizer()
        visualizer.add_machine_type(machine_type)

        model = WaveNet(n_blocks=args.n_blocks,
                        n_channel=args.n_mels,
                        n_mul=args.channel_mul,
                        kernel_size=args.kernel_size,
                        padding='causal',
                        n_groups=args.n_groups)

        training_data, val_data = dataset.load_wavenet_dataset(args, target_dir)
        train_loss = tf.keras.metrics.Mean(name='train loss')

        wn_trainer = WaveNetTrainer(args=args,
                                 machine_type=machine_type,
                                 visualizer=visualizer,
                                 model=model,
                                 train_loss=train_loss)
        mean, inv_cov = wn_trainer.train(training_data, val_data)
        mean_list.append(mean)
        inv_cov_list.append(inv_cov)
    return mean_list, inv_cov_list


def test_wavenet(mean_list, inv_cov_list):
    for idx, machine_type in enumerate(args.machines):
        mean, inv_cov = mean_list[idx], inv_cov_list[idx]
        print('\n' + '=' * 60)
        print(f'[{idx + 1}/{len(args.machines)}] {machine_type}')
        model_path = f'{args.model_dir}/{args.version}/{machine_type}/checkpoint_best_model'

        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(model_path)
        wn_trainer = WaveNetTrainer(args=args,
                                 machine_type=machine_type,
                                 visualizer=None,
                                 model=model,
                                 train_loss=None)
        wn_trainer.test(mean, inv_cov)
    return


def train_resnet():
    dirs = utils.select_dirs(args)
    for idx, target_dir in enumerate(dirs):
        print('\n' + '=' * 60)
        print(f'[{idx + 1}/{len(dirs)}] {target_dir}')

        machine_type = os.path.split(target_dir)[1]

        visualizer = WaveNetVisualizer()
        visualizer.add_machine_type(machine_type)

        n_class = 6 if machine_type == 'ToyConveyor' else 7
        model = tf.keras.applications.resnet50.ResNet50(weights=None,
                                                        input_shape=(args.n_mels, args.frames, 1),
                                                        classes=n_class)

        training_data, val_data = dataset.load_resnet_dataset(args, target_dir)
        train_loss = tf.keras.metrics.Mean(name='train loss')

        rn_trainer = ResNetTrainer(args=args,
                                   machine_type=machine_type,
                                   visualizer=visualizer,
                                   model=model,
                                   train_loss=train_loss,
                                   n_class=n_class)

        rn_trainer.train(training_data, val_data)


def test_resnet():
    for idx, machine_type in enumerate(args.machines):
        print('\n' + '=' * 60)
        print(f'[{idx + 1}/{len(args.machines)}] {machine_type}')
        model_path = f'{args.model_dir}/{args.version}/{machine_type}/checkpoint_best_model'
        n_class = 6 if machine_type == 'ToyConveyor' else 7
        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(model_path)
        rn_trainer = ResNetTrainer(args=args,
                                   machine_type=machine_type,
                                   visualizer=None,
                                   model=model,
                                   train_loss=None,
                                   n_class=n_class)
        rn_trainer.test()


def train_mtl_class():
    dirs = utils.select_dirs(args)
    mean_list, inv_cov_list, score_mean_list, score_inv_cov_list = [], [], [], []
    for idx, target_dir in enumerate(dirs):
        print('\n' + '=' * 60)
        print(f'[{idx + 1}/{len(dirs)}] {target_dir}')

        machine_type = os.path.split(target_dir)[1]

        visualizer = MTLClassVisualizer()
        visualizer.add_machine_type(machine_type)

        n_class = 6 if machine_type == 'ToyConveyor' else 7
        model = MTLClass(n_blocks=args.n_blocks,
                         n_channel=args.n_mels,
                         n_mul=args.channel_mul,
                         kernel_size=args.kernel_size,
                         padding='causal',
                         n_groups=args.n_groups,
                         n_class=n_class,
                         t_frame=args.frames,
                         arcface=ArcMarginProduct())

        training_data, val_data = dataset.load_mtl_class_dataset(args, target_dir)
        train_loss = tf.keras.metrics.Mean(name='train loss')
        train_loss1 = tf.keras.metrics.Mean(name='train loss 1')
        train_loss2 = tf.keras.metrics.Mean(name='train loss 2')

        mtl_trainer = MTLClassTrainer(args=args,
                                      machine_type=machine_type,
                                      visualizer=visualizer,
                                      model=model,
                                      train_loss=train_loss,
                                      train_loss1=train_loss1,
                                      train_loss2=train_loss2,
                                      n_class=n_class)

        mean, inv_cov, score_mean, score_inv_cov = mtl_trainer.train(training_data, val_data)

        mean_list.append(mean)
        inv_cov_list.append(inv_cov)
        score_mean_list.append(score_mean)
        score_inv_cov_list.append(score_inv_cov)
    return mean_list, inv_cov_list, score_mean_list, score_inv_cov_list


def test_mtl_class(mean_list, inv_cov_list, score_mean_list, score_inv_cov_list):
    for idx, machine_type in enumerate(args.machines):
        mean, inv_cov, score_mean, score_inv_cov = mean_list[idx], inv_cov_list[idx], score_mean_list[idx], score_inv_cov_list[idx]
        print('\n' + '=' * 60)
        print(f'[{idx + 1}/{len(args.machines)}] {machine_type}')
        model_path = f'{args.model_dir}/{args.version}/{machine_type}/checkpoint_best_model'

        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(model_path)
        n_class = 6 if machine_type == 'ToyConveyor' else 7

        mtl_trainer = MTLClassTrainer(args=args,
                                      machine_type=machine_type,
                                      visualizer=None,
                                      model=model,
                                      train_loss=None,
                                      train_loss1=None,
                                      train_loss2=None,
                                      n_class=n_class)
        mtl_trainer.test(mean, inv_cov, score_mean, score_inv_cov)


def train_mrwn(is_sum=False):
    dirs = utils.select_dirs(args)
    mean_list, inv_cov_list, block_mean_list, block_inv_cov_list = [], [], [], []
    for idx, target_dir in enumerate(dirs):
        print('\n' + '=' * 60)
        print(f'[{idx + 1}/{len(dirs)}] {target_dir}')

        machine_type = os.path.split(target_dir)[1]

        visualizer = WaveNetVisualizer()
        visualizer.add_machine_type(machine_type)

        if is_sum:
            model = MultiResolutionSumWaveNet(n_blocks=args.n_blocks,
                                              n_channel=args.n_mels,
                                              n_mul=args.channel_mul,
                                              kernel_size=args.kernel_size,
                                              padding='causal',
                                              n_groups=args.n_groups)
        else:
            model = MultiResolutionWaveNet(n_blocks=args.n_blocks,
                                           n_channel=args.n_mels,
                                           n_mul=args.channel_mul,
                                           kernel_size=args.kernel_size,
                                           padding='causal',
                                           n_groups=args.n_groups)

        training_data, val_data = dataset.load_wavenet_dataset(args, target_dir)
        train_loss = tf.keras.metrics.Mean(name='train loss')

        mrwn_trainer = MultiResolutionWaveNetTrainer(args=args,
                                                     machine_type=machine_type,
                                                     visualizer=visualizer,
                                                     model=model,
                                                     train_loss=train_loss)

        mean, inv_cov, block_mean, block_inv_cov = mrwn_trainer.train(training_data, val_data)
        mean_list.append(mean)
        inv_cov_list.append(inv_cov)
        block_mean_list.append(block_mean)
        block_inv_cov_list.append(block_inv_cov)
    return mean_list, inv_cov_list, block_mean_list, block_inv_cov_list


def test_mrwn(mean_list, inv_cov_list, block_mean_list, block_inv_cov_list):
    for idx, machine_type in enumerate(args.machines):
        mean, inv_cov, block_mean, block_inv_cov = mean_list[idx], inv_cov_list[idx], block_mean_list[idx], block_inv_cov_list[idx]
        print('\n' + '=' * 60)
        print(f'[{idx + 1}/{len(args.machines)}] {machine_type}')
        model_path = f'{args.model_dir}/{args.version}/{machine_type}/checkpoint_best_model'

        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(model_path)
        mrwn_trainer = MultiResolutionWaveNetTrainer(args=args,
                                                     machine_type=machine_type,
                                                     visualizer=None,
                                                     model=model,
                                                     train_loss=None)
        mrwn_trainer.test(mean, inv_cov, block_mean, block_inv_cov)


def train_mtl_seg():
    dirs = utils.select_dirs(args)
    mean_list, inv_cov_list, score_mean_list, score_inv_cov_list = [], [], [], []
    for idx, target_dir in enumerate(dirs):
        print('\n' + '=' * 60)
        print(f'[{idx + 1}/{len(dirs)}] {target_dir}')

        machine_type = os.path.split(target_dir)[1]

        visualizer = MTLClassVisualizer()
        visualizer.add_machine_type(machine_type)

        n_class = 6 if machine_type == 'ToyConveyor' else 7

        model = MTLSeg(n_blocks=args.n_blocks,
                       n_channel=args.n_mels,
                       n_mul=args.channel_mul,
                       kernel_size=args.kernel_size,
                       padding='causal',
                       n_groups=args.n_groups,
                       n_class=n_class)

        training_data, val_data = dataset.load_mtl_class_dataset(args, target_dir, is_seg=True)
        train_loss = tf.keras.metrics.Mean(name='train loss')
        train_loss1 = tf.keras.metrics.Mean(name='train loss 1')
        train_loss2 = tf.keras.metrics.Mean(name='train loss 2')

        mtl_trainer = MTLSegmentationTrainer(args=args,
                                             machine_type=machine_type,
                                             visualizer=visualizer,
                                             model=model,
                                             train_loss=train_loss,
                                             train_loss1=train_loss1,
                                             train_loss2=train_loss2,
                                             n_class=n_class)

        mean, inv_cov, score_mean, score_inv_cov = mtl_trainer.train(training_data, val_data)

        mean_list.append(mean)
        inv_cov_list.append(inv_cov)
        score_mean_list.append(score_mean)
        score_inv_cov_list.append(score_inv_cov)
    return mean_list, inv_cov_list, score_mean_list, score_inv_cov_list


def test_mtl_seg(mean_list, inv_cov_list, score_mean_list, score_inv_cov_list):
    for idx, machine_type in enumerate(args.machines):
        mean, inv_cov, score_mean, score_inv_cov = mean_list[idx], inv_cov_list[idx], score_mean_list[idx], score_inv_cov_list[idx]
        print('\n' + '=' * 60)
        print(f'[{idx + 1}/{len(args.machines)}] {machine_type}')
        model_path = f'{args.model_dir}/{args.version}/{machine_type}/checkpoint_best_model'

        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(model_path)
        n_class = 6 if machine_type == 'ToyConveyor' else 7

        mtl_trainer = MTLSegmentationTrainer(args=args,
                                             machine_type=machine_type,
                                             visualizer=None,
                                             model=model,
                                             train_loss=None,
                                             train_loss1=None,
                                             train_loss2=None,
                                             n_class=n_class)
        mtl_trainer.test(mean, inv_cov, score_mean, score_inv_cov)


def train_mtl_class_seg():
    dirs = utils.select_dirs(args)
    mean_list, inv_cov_list, score_mean_list, score_inv_cov_list = [], [], [], []
    for idx, target_dir in enumerate(dirs):
        print('\n' + '=' * 60)
        print(f'[{idx + 1}/{len(dirs)}] {target_dir}')

        machine_type = os.path.split(target_dir)[1]

        visualizer = MTLClassSegVisualizer()
        visualizer.add_machine_type(machine_type)

        n_class = 6 if machine_type == 'ToyConveyor' else 7

        model = MTLClassSeg(n_blocks=args.n_blocks,
                            n_channel=args.n_mels,
                            n_mul=args.channel_mul,
                            kernel_size=args.kernel_size,
                            padding='causal',
                            n_groups=args.n_groups,
                            n_class=n_class,
                            t_frame=args.frames,
                            arcface=None)

        training_data, val_data = dataset.load_mtl_class_seg_dataset(args, target_dir)
        train_loss = tf.keras.metrics.Mean(name='train loss')
        train_loss1 = tf.keras.metrics.Mean(name='train loss 1')
        train_loss2 = tf.keras.metrics.Mean(name='train loss 2')
        train_loss3 = tf.keras.metrics.Mean(name='train loss 3')

        mtl_trainer = MTLClassSegTrainer(args=args,
                                         machine_type=machine_type,
                                         visualizer=visualizer,
                                         model=model,
                                         train_loss=train_loss,
                                         train_loss1=train_loss1,
                                         train_loss2=train_loss2,
                                         train_loss3=train_loss3,
                                         n_class=n_class)

        mean, inv_cov, score_mean, score_inv_cov = mtl_trainer.train(training_data, val_data)

        mean_list.append(mean)
        inv_cov_list.append(inv_cov)
        score_mean_list.append(score_mean)
        score_inv_cov_list.append(score_inv_cov)
    return mean_list, inv_cov_list, score_mean_list, score_inv_cov_list

def test_mtl_class_seg(mean_list, inv_cov_list, score_mean_list, score_inv_cov_list):
    for idx, machine_type in enumerate(args.machines):
        mean, inv_cov, score_mean, score_inv_cov = mean_list[idx], inv_cov_list[idx], score_mean_list[idx], score_inv_cov_list[idx]
        print('\n' + '=' * 60)
        print(f'[{idx + 1}/{len(args.machines)}] {machine_type}')
        model_path = f'{args.model_dir}/{args.version}/{machine_type}/checkpoint_best_model'

        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(model_path)
        n_class = 6 if machine_type == 'ToyConveyor' else 7

        mtl_trainer = MTLClassSegTrainer(args=args,
                                         machine_type=machine_type,
                                         visualizer=None,
                                         model=model,
                                         train_loss=None,
                                         train_loss1=None,
                                         train_loss2=None,
                                         train_loss3=None,
                                         n_class=n_class)

        mtl_trainer.test(mean, inv_cov, score_mean, score_inv_cov)


def main(args):
    preprocess()

    if args.training_mode == 'WaveNet':
        mean_list, inv_cov_list = train_wavenet()
        test_wavenet(mean_list, inv_cov_list)

    elif args.training_mode == 'ResNet50':
        train_resnet()
        test_resnet()

    elif args.training_mode == 'MTL_class':
        mean_list, inv_cov_list, score_mean_list, score_inv_cov_list = train_mtl_class()
        test_mtl_class(mean_list, inv_cov_list, score_mean_list, score_inv_cov_list)

    elif args.training_mode == 'MTL_seg':
        mean_list, inv_cov_list, score_mean_list, score_inv_cov_list = train_mtl_seg()
        test_mtl_seg(mean_list, inv_cov_list, score_mean_list, score_inv_cov_list)

    elif args.training_mode == 'MRWN':
        mean_list, inv_cov_list, block_mean_list, block_inv_cov_list = train_mrwn(is_sum=False)
        test_mrwn(mean_list, inv_cov_list, block_mean_list, block_inv_cov_list)

    elif args.training_mode == 'MRSWN':
        mean_list, inv_cov_list, block_mean_list, block_inv_cov_list = train_mrwn(is_sum=True)
        test_mrwn(mean_list, inv_cov_list, block_mean_list, block_inv_cov_list)

    elif args.training_mode == 'MTL_class_seg':
        mean_list, inv_cov_list, score_mean_list, score_inv_cov_list = train_mtl_class_seg()
        test_mtl_class_seg(mean_list, inv_cov_list, score_mean_list, score_inv_cov_list)



if __name__ == "__main__":
    args = parser.parse_args()
    set_random_everything(param['seed'])
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(v) for v in args.device_ids])
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    print(f'Model path: {args.version}')
    main(args)