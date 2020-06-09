import os
import argparse
from solver import Solver
from subsample import subsample
from data_loader import get_loader
from torch.backends import cudnn
from itertools import product
from train_on_fake import train_on_fake
from gan_test import test_train_on_fake

def generate_sensible_labels(selected_attrs):
    hair_color_indices = []
    for i, attr_name in enumerate(selected_attrs):
        if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
            hair_color_indices.append(i)

    labels_dict = {}
    label_id = 0
    for c_trg in product(*[[-1, 1]] * len(selected_attrs)):
        hair_color_sublabel = [c_trg[i] for i in hair_color_indices]
        if sum(hair_color_sublabel) > -1:
            continue
        else:
            labels_dict[label_id] = list(c_trg)
            label_id += 1
    return labels_dict

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader.
    labelled_loader = None
    unlabelled_loader = None

    if config.dataset == 'CelebA':
        if config.mode == 'train':
            # Read CelebA list file and split training data into labelled and unlabelled pools.
            labels_dict = generate_sensible_labels(config.selected_attrs)
            subsample(labelled_percentage=config.labelled_percentage,
                      selected_attrs=config.selected_attrs,
                      celeba_dir=config.celeba_dir,
                      labels_dict=labels_dict)

            labelled_loader = {}
            for cls in labels_dict:
                labelled_loader[cls] = get_loader(config.celeba_image_dir,
                                                  config.attr_path,
                                                  config.selected_attrs,
                                                  True,
                                                  config.celeba_crop_size,
                                                  config.image_size,
                                                  4,
                                                  'CelebA',
                                                  'train',
                                                  config.num_workers,
                                                  cls)

            unlabelled_loader = get_loader(config.celeba_image_dir,
                                           config.attr_path,
                                           config.selected_attrs,
                                           True,
                                           config.celeba_crop_size,
                                           config.image_size,
                                           config.batch_size,
                                           'CelebA',
                                           'unlabelled',
                                           config.num_workers)
        else:
            if config.mode == 'synth':
                config.batch_size = 256
            labelled_loader = get_loader(config.celeba_image_dir,
                                         config.attr_path,
                                         config.selected_attrs,
                                         False,
                                         config.celeba_crop_size,
                                         config.image_size,
                                         config.batch_size,
                                         'CelebA',
                                         config.mode,
                                         config.num_workers)

    elif config.dataset == 'RaFD':
        if config.mode == 'train':
            labelled_loader = {}
            for cls in range(8):
                labelled_loader[cls] = get_loader(config.rafd_image_dir,
                                                  None,
                                                  None,
                                                  True,
                                                  config.rafd_crop_size,
                                                  config.image_size,
                                                  4,
                                                  'RaFD',
                                                  'train',
                                                  config.num_workers,
                                                  cls,
                                                  pc=config.labelled_percentage)
            unlabelled_loader = get_loader(config.rafd_image_dir,
                                           None,
                                           None,
                                           True,
                                           config.rafd_crop_size,
                                           config.image_size,
                                           config.batch_size,
                                           'RaFD',
                                           'unlabelled',
                                           config.num_workers,
                                           pc=config.labelled_percentage)
        else:
            if config.mode == 'synth':
                config.batch_size = 256
                rafd_shuffle = True
            else:
                rafd_shuffle = False
            labelled_loader = get_loader(config.rafd_image_dir,
                                         None,
                                         None,
                                         rafd_shuffle,
                                         config.rafd_crop_size,
                                         config.image_size,
                                         config.batch_size,
                                         'RaFD',
                                         config.mode,
                                         config.num_workers)

    # Solver for training, testing, and evaluating MatchGAN.
    solver = Solver(labelled_loader, unlabelled_loader, config)

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()
    elif config.mode == 'eval':
        solver.get_fid()
        solver.get_inception_score()
        solver.attr_cls()
    elif config.mode == 'synth':
        if not os.path.isdir(os.path.join(config.result_dir, 'synthetic_tfrecord')):
            solver.generate_synthetic_data()
        train_on_fake(config.dataset, config.c_dim, config.result_dir, config.device)
    elif config.mode == 'synth_test': #GAN-train accuracy
        test_train_on_fake(config.dataset, config.c_dim, config.result_dir, config.device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--c2_dim', type=int, default=8, help='dimension of domain labels (2nd dataset)')
    parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
    parser.add_argument('--rafd_crop_size', type=int, default=256, help='crop size for the RaFD dataset')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--lambda_match', type=float, default=0.5, help='weight for match loss')

    # Training configuration.
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'RaFD', 'Both'])
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])
    parser.add_argument('--selected_emots', nargs='+', help='selected emotions for the RaFD dataset',
                        default=['angry', 'contemptuous', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised'])

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'eval', 'synth', 'synth_test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    parser.add_argument('--synth_seed', type=int, default=12345)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--labelled_percentage', type=int, default=100)

    # Directories.
    parser.add_argument('--celeba_dir', type=str, default='./data/celeba')
    parser.add_argument('--celeba_image_dir', type=str, default='./data/celeba/images')
    parser.add_argument('--attr_path', type=str, default='./data/celeba/list_attr_celeba.txt')
    parser.add_argument('--rafd_image_dir', type=str, default='./data/RaFD/preprocessed')
    parser.add_argument('--log_dir', type=str, default='outputs/logs')
    parser.add_argument('--model_save_dir', type=str, default='outputs/models')
    parser.add_argument('--sample_dir', type=str, default='outputs/samples')
    parser.add_argument('--result_dir', type=str, default='outputs/results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=5)
    parser.add_argument('--print_samples', type=int, default=0, choices=[0, 1])
    parser.add_argument('--sample_step', type=int, default=100)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    print(config)
    main(config)
