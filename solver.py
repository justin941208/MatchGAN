import os
import time
import datetime
import pathlib
import torch
import torch.nn.functional as F
import numpy as np
import tensorflow as tf
import models
import tflib as tl
from PIL import Image
from glob import glob
from itertools import product
from model import Generator, Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
from fid_score import calculate_fid_given_paths
from inception_score import inception_score, FakeImageDataset

def arr_2_str(array, delimiter=','):
    if len(array.shape) > 0:
        return delimiter.join(array.astype('str'))
    else:
        return array.astype('str')

class Solver(object):
    """Solver for training, testing, and evaluating MatchGAN."""

    def __init__(self, labelled_loader, unlabelled_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.labelled_loader = labelled_loader
        self.unlabelled_loader = unlabelled_loader

        # Model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.lambda_match = config.lambda_match

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs
        self.selected_emots = config.selected_emots

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda:{}'.format(config.device))
        self.device_num = config.device
        self.all_target_labels = self.generate_target_labels(5, config.selected_attrs)
        self.synth_seed = config.synth_seed
        self.mode = config.mode

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        torch.manual_seed(1234)
        if self.dataset in ['CelebA', 'RaFD']:
            self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        if self.mode == 'train':
            D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
            self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                if i in hair_color_indices:
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)
            elif dataset == 'RaFD':
                c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        if dataset == 'CelebA':
            return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
        elif dataset == 'RaFD':
            return F.cross_entropy(logit, target)

    def generate_target_labels(self, c_dim=5, selected_attrs=None):
        hair_color_indices = []
        for i, attr_name in enumerate(selected_attrs):
            if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                hair_color_indices.append(i)

        labels = torch.zeros(0, c_dim)
        for c_trg in product(*[[0.0, 1.0]] * c_dim):
            hair_color_sublabel = [c_trg[i] for i in hair_color_indices]
            if np.sum(hair_color_sublabel) > 1:
                continue
            else:
                labels = torch.cat([labels, torch.Tensor([c_trg])])

        return labels

    def get_pairs(self, x, label=None, triple=False):
        gp_1 = [i for i in range(x.size(0)) if i % 4 == 0]
        gp_2 = [i + 1 for i in gp_1]
        gp_3 = [i + 2 for i in gp_1]
        gp_4 = [i + 3 for i in gp_1]
        gp_4 = gp_4[-1:] + gp_4[:-1]

        x_pair_1 = x[gp_1 + gp_3]
        x_pair_2 = x[gp_2 + gp_4]

        stop_num = int(x_pair_1.size(0) / 2)
        labels_pair = torch.ones(stop_num)
        labels_pair = torch.cat([labels_pair, torch.zeros(stop_num)]).to(torch.long)
        labels_pair = labels_pair.to(self.device)

        if triple:
            gp_4_ = gp_4[-1:] + gp_4[:-1]
            x_pair_1 = x[gp_1 + gp_3 + gp_1 + gp_3]
            x_pair_2 = x[gp_2 + gp_4 + gp_4_ + gp_2]
            labels_pair = torch.cat([labels_pair, labels_pair[stop_num:], labels_pair[:stop_num]])

        if label is not None:
            if triple:
                label_pair_1 = label[gp_1 + gp_3 + gp_1 + gp_3]
                label_pair_2 = label[gp_2 + gp_4 + gp_4_ + gp_2]
            else:
                label_pair_1 = label[gp_1 + gp_3]
                label_pair_2 = label[gp_2 + gp_4]
            labels_diff = label_pair_2 - label_pair_1
            return x_pair_1, x_pair_2, labels_pair, labels_diff
        else:
            return x_pair_1, x_pair_2, labels_pair

    def train(self):
        """Train MatchGAN within a single dataset."""
        # Set seed.
        np.random.seed(1234)

        # Set data loader.
        labelled_loader = self.labelled_loader
        unlabelled_loader = self.unlabelled_loader

        labelled_iters_dict = {}
        for cls in labelled_loader:
            labelled_iters_dict[cls] = iter(labelled_loader[cls])
        if unlabelled_loader is not None:
            unlabelled_iter = iter(unlabelled_loader)
        classes = [cls for cls in labelled_loader]
        trg_classes = classes.copy()
        np.random.shuffle(classes)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()

        for i in range(start_iters, self.num_iters):

            # Flag for labelled/unlabelled data.
            if unlabelled_loader is not None:
                label_flag = True if i % 2 == 0 else False
            else:
                label_flag = True

            # A random sample of the attributes to select data from.
            if len(classes) > 0:
                cls_sample = classes[:4]
                classes = classes[4:]
            else:
                classes = [cls for cls in labelled_loader]
                np.random.shuffle(classes)
                cls_sample = classes[:4]
                classes = classes[4:]

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Get next batch.
            if label_flag:
                x_real_labelled_list = []
                label_org_list = []
                for cls in cls_sample:
                    try:
                        x_r, l_o = next(labelled_iters_dict[cls])
                    except:
                        labelled_iters_dict[cls] = iter(labelled_loader[cls])
                        x_r, l_o = next(labelled_iters_dict[cls])
                    x_real_labelled_list.append(x_r)
                    label_org_list.append(l_o)
                x_real = torch.cat(x_real_labelled_list)
                label_org = torch.cat(label_org_list)
            else:
                try:
                    x_real, _ = next(unlabelled_iter)
                except:
                    unlabelled_iter = iter(unlabelled_loader)
                    x_real, _ = next(unlabelled_iter)

            # Generate target domain labels randomly.
            if self.dataset == 'CelebA':
                sample = np.random.choice(trg_classes, size=4, replace=False).repeat(4)
                shfl_idx = torch.cat([torch.arange(4) * 4 + i for i in range(4)])
                rvrs_idx = shfl_idx
                label_trg = self.all_target_labels[sample][shfl_idx]
            elif self.dataset == 'RaFD':
                sample = np.random.choice(np.arange(8), size=4, replace=False).repeat(4)
                shfl_idx = torch.cat([torch.arange(4) * 4 + i for i in range(4)])
                rvrs_idx = shfl_idx
                label_trg = torch.arange(8)[sample][shfl_idx]

            if self.dataset == 'CelebA':
                c_org = label_org.clone()
                c_trg = label_trg.clone()

            elif self.dataset == 'RaFD':
                c_org = self.label2onehot(label_org, self.c_dim)
                c_trg = self.label2onehot(label_trg, self.c_dim)

            x_real = x_real.to(self.device)           # Input images.
            c_org = c_org.to(self.device)             # Original domain labels.
            c_trg = c_trg.to(self.device)             # Target domain labels.
            label_org = label_org.to(self.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.

            # =================================================================================== #
            #                               2. Train the discriminator                            #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls = self.D(x_real, False)
            d_loss_real = -torch.mean(out_src)
            if label_flag:
                d_loss_cls = self.classification_loss(out_cls, label_org, self.dataset)

            # Compute loss with fake images.
            x_fake = self.G(x_real, c_trg)
            out_src, out_cls = self.D(x_fake.detach(), False)
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat, False)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Compute match loss.
            if label_flag:
                x_r_1, x_r_2, labels_pair_r = self.get_pairs(x_real, triple=True)
                out_pairs_r = self.D([x_r_1, x_r_2], True)
                d_loss_match = F.cross_entropy(out_pairs_r, labels_pair_r)

            # Backward and optimize.
            if label_flag:
                d_loss = d_loss_real + \
                         d_loss_fake + \
                         self.lambda_cls * d_loss_cls + \
                         self.lambda_gp * d_loss_gp + \
                         self.lambda_match * d_loss_match
            else:
                d_loss = d_loss_real + \
                         d_loss_fake + \
                         self.lambda_gp * d_loss_gp

            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            loss['D/loss_match'] = d_loss_match.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                x_fake = self.G(x_real, c_trg)
                out_src, out_cls = self.D(x_fake, False)
                g_loss_fake = -torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg, self.dataset)

                # Target-to-original domain.
                if label_flag:
                    x_reconst = self.G(x_fake, c_org)
                    g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # Compute match loss.
                x_f_1, x_f_2, labels_pair_f = self.get_pairs(x_fake[rvrs_idx], triple=True)
                out_pairs_f = self.D([x_f_1, x_f_2], True)
                g_loss_match = F.cross_entropy(out_pairs_f, labels_pair_f)

                # Backward and optimize.
                if label_flag:
                    g_loss = g_loss_fake + \
                             self.lambda_cls * g_loss_cls + \
                             self.lambda_rec * g_loss_rec + \
                             self.lambda_match * g_loss_match
                else:
                    g_loss = g_loss_fake + \
                             self.lambda_cls * g_loss_cls + \
                             self.lambda_match * g_loss_match

                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()
                loss['G/loss_match'] = g_loss_match.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def test(self, num_batches=None):
        """Translate images using MatchGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)

        # Set data loader.
        data_loader = self.labelled_loader

        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):

                if num_batches is not None:
                    if i == num_batches:
                        break

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

                # Translate images.
                x_fake_list = [x_real]
                for c_trg in c_trg_list:
                    x_fake_list.append(self.G(x_real, c_trg))

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                filename = '{}-images.jpg'.format(i+1)
                result_path = os.path.join(self.result_dir, filename)
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))

    def get_fid(self):
        """Calculate the FID score"""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        data_loader = self.labelled_loader

        if self.dataset == 'CelebA':
            attr_list = self.selected_attrs
        elif self.dataset == 'RaFD':
            attr_list = self.selected_emots

        real_dir = os.path.join(self.result_dir, 'real')
        if not os.path.isdir(real_dir):
            os.mkdir(real_dir)
        fake_dir = os.path.join(self.result_dir, 'fake_all')
        if not os.path.isdir(fake_dir):
            os.mkdir(fake_dir)

            with torch.no_grad():
                r_count = 0
                f_count = 0
                for i, (x_real, c_org) in enumerate(data_loader):

                    # Prepare input images and target domain labels.
                    x_real = x_real.to(self.device)
                    c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)
                    if self.dataset == 'RaFD':
                        c_org = self.label2onehot(c_org, self.c_dim)

                    # Save real images.
                    for r in range(x_real.size(0)):
                        r_count += 1
                        c_o = ''.join(c_org[r].numpy().astype('int8').astype('str'))
                        real_path = os.path.join(real_dir, 'r_{}_{}.jpg'.format(r_count, c_o))
                        save_image(self.denorm(x_real[r].data.cpu()), real_path)

                    # Translate real images into fake ones and save them.
                    f_count_old = f_count
                    for c_trg, attr_name in zip(c_trg_list, attr_list):
                        x_fake = self.G(x_real, c_trg)
                        if f_count > f_count_old:
                            f_count = f_count_old
                        for f in range(x_fake.size(0)):
                            f_count += 1
                            c_t = ''.join(c_trg[f].cpu().numpy().astype('int8').astype('str'))
                            fake_path = os.path.join(fake_dir, attr_name + '_{}_{}.jpg'.format(f_count, c_t))
                            save_image(self.denorm(x_fake[f].data.cpu()), fake_path)

        fids = []
        fid_path = os.path.join(self.result_dir, 'fid_scores.txt')
        fake_paths_list = []
        for attr_name in attr_list:
            fake_paths_list.append(list(pathlib.Path(fake_dir).glob('{}*'.format(attr_name))))
        fake_paths_list.append(fake_dir)
        fids = calculate_fid_given_paths([real_dir, fake_paths_list], 128, True, 2048, device=self.device_num)
        with open(fid_path, 'w') as f:
            for attr_name, fid in zip(attr_list, fids[:-1]):
                f.write('{}: {:.4f}\n'.format(attr_name, fid))
            f.write('Average FID: {:.4f}\n'.format(np.mean(fids[:-1])))
            f.write('Overall FID: {:.4f}\n'.format(fids[-1]))
        print('Overall FID: {:.4f}\n'.format(fids[-1]))

    def get_inception_score(self):
        fake_dataset = FakeImageDataset(self.result_dir)

        print("Calculating Inception Score...")
        m, std = inception_score(fake_dataset, cuda=True, batch_size=32, resize=True, splits=10, device=self.device_num)
        print(m, std)

        file_path = os.path.join(self.result_dir, 'inception_score.txt')
        with open(file_path, 'w') as f:
            f.write('Inception score: {:.4f}, standard deviation: {:.4f}'.format(m, std))

    def attr_cls(self):
        """Computes the GAN-test attribute classification accuracy."""
        # Load the trained generator.
        self.restore_model(self.test_iters)

        # Set data loader.
        data_loader = self.labelled_loader

        attr_list = []
        if self.dataset == 'CelebA':
            ckpt_file = './checkpoints_train_on_real/CelebA/Epoch_(127)_(2543of2543).ckpt'
            attr_list = self.selected_attrs
            n_print = 2000
        elif self.dataset == 'RaFD':
            ckpt_file = './checkpoints_train_on_real/RaFD/Epoch_(199)_(112of112).ckpt'
            attr_list = self.selected_emots
            n_print = 200
        classifier = models.classifier

        # Classifier graph
        x = tf.placeholder(tf.float32, shape=[None, 128, 128, 3])
        logits = classifier(x, att_dim=len(attr_list), reuse=False, training=False)
        if self.dataset == 'CelebA':
            pred_s = tf.cast(tf.nn.sigmoid(logits), tf.float64)
        elif self.dataset == 'RaFD':
            pred_s = tf.cast(tf.nn.softmax(logits), tf.float64)

        cnt_pos = np.zeros([self.c_dim]).astype(np.int64)
        cnt_neg = np.zeros([self.c_dim]).astype(np.int64)
        cnt_rec = np.zeros([self.c_dim]).astype(np.int64)
        c_pos = np.zeros([self.c_dim])
        c_neg = np.zeros([self.c_dim])
        c_rec = np.zeros([self.c_dim])
        ca_req = np.zeros([self.c_dim]).astype(np.int64)
        cr_req = np.zeros([self.c_dim]).astype(np.int64)
        co_req = np.zeros([self.c_dim]).astype(np.int64)

        with torch.no_grad():
            with tl.session() as sess:
                tl.load_checkpoint(ckpt_file, sess)
                attr_list = ['Reconstruction'] + attr_list
                total_count = 0
                for i, (x_real, c_org) in enumerate(data_loader):

                    if self.dataset == 'RaFD':
                        c_org = self.label2onehot(c_org, self.c_dim)

                    # Prepare input images and target domain labels.
                    x_real = x_real.to(self.device)
                    c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)
                    c_trg_batch = torch.cat([c.unsqueeze(1) for c in c_trg_list], dim=1).cpu().numpy()
                    c_trg_list = [None] + [c_org.to(self.device)] + c_trg_list
                    att_gt_batch = c_org.numpy()

                    # Classify translate images.
                    pred_score_list =[]
                    preds_list = []
                    for j, c_trg in enumerate(c_trg_list):
                        if j == 0:
                            feed = np.transpose(x_real.cpu().numpy(), [0, 2, 3, 1])
                        else:
                            x_fake = self.G(x_real, c_trg)
                            feed = np.transpose(x_fake.cpu().numpy(), [0, 2, 3, 1])
                        pred_score = sess.run(pred_s, feed_dict={x: feed})
                        pred_score_list.append(np.expand_dims(pred_score, axis=1))
                        if self.dataset == 'CelebA':
                            preds = np.round(pred_score).astype(int)
                        elif self.dataset == 'RaFD':
                            max_id = np.argmax(pred_score, axis=1)
                            preds = np.zeros_like(pred_score).astype(int)
                            preds[np.arange(pred_score.shape[0]), max_id] = 1
                        preds_list.append(np.expand_dims(preds, axis=1))
                    pred_score_batch = np.concatenate(pred_score_list, axis=1)
                    preds_opt_batch = np.concatenate(preds_list, axis=1)

                    # Calculate accuracy.
                    for pred_score, preds_opt, att_gt, c_trg in zip(pred_score_batch, preds_opt_batch, att_gt_batch, c_trg_batch):
                        for k in range(2, len(preds_opt)):
                            if c_trg[k - 2, k - 2] == 1 - att_gt[k - 2]:
                                if att_gt[k - 2] == 0:
                                    ca_req[k-2] += 1
                                elif att_gt[k - 2] == 1:
                                    cr_req[k-2] += 1

                                if preds_opt[k, k - 2] == 1 - att_gt[k - 2]:
                                    if preds_opt[k, k - 2] == 1:
                                        cnt_pos[k-2] += 1
                                        c_pos[k-2] += pred_score[k, k - 2]
                                    elif preds_opt[k, k - 2] == 0:
                                        cnt_neg[k-2] += 1
                                        c_neg[k-2] += 1 - pred_score[k, k - 2]
                            else:
                                co_req[k-2] += 1
                                if preds_opt[k, k - 2] == att_gt[k - 2]:
                                    cnt_rec[k-2] += 1
                                    if preds_opt[k, k - 2] == 1:
                                        c_rec[k-2] += pred_score[k, k - 2]
                                    elif preds_opt[k, k - 2] == 0:
                                        c_rec[k-2] += 1 - pred_score[k, k - 2]

                    total_count += x_real.shape[0]
                    if total_count % n_print == 0:
                        print('{} images classified.'.format(total_count))
                        print('\tAcc. Addition')
                        print('\t', cnt_pos / ca_req)
                        print('\t', np.mean(cnt_pos / ca_req))

                attr_cls_path = os.path.join(self.result_dir, 'GAN-test.txt')
                with open(attr_cls_path, 'w') as f:
                    f.write('Overall accuracy,{},average,{}\n'.format(arr_2_str((cnt_pos + cnt_neg + cnt_rec) / (ca_req + cr_req + co_req)),
                                                                      arr_2_str(np.mean((cnt_pos + cnt_neg + cnt_rec) / (ca_req + cr_req + co_req)))))
        print('GAN-test accuracy: {}'.format(arr_2_str(np.mean((cnt_pos + cnt_neg + cnt_rec) / (ca_req + cr_req + co_req)))))

    def generate_synthetic_data(self):
        """Generates synthetic images for training classifiers."""
        # Load the trained generator.
        torch.manual_seed(self.synth_seed)
        self.restore_model(self.test_iters)

        # Set data loader.
        data_loader = self.labelled_loader

        out_dir = os.path.join(self.result_dir, 'synthetic_tfrecord')
        if not os.path.isdir(out_dir):

            # create a writer
            writer = tl.tfrecord.ImageLablePairTfrecordCreator(save_path=out_dir,
                                                               encode_type=None,
                                                               data_name='img',
                                                               compression_type=0)

            with torch.no_grad():
                f_count = 0
                for i, (x_real, c_org) in enumerate(data_loader):

                    # Prepare input images and target domain labels.
                    x_real = x_real.to(self.device)
                    rand_idx = torch.randperm(c_org.size(0))
                    c_trg = c_org[rand_idx]

                    # Translate real images into fake ones.
                    if self.dataset == 'CelebA':
                        x_fake = self.denorm(self.G(x_real, c_trg.to(self.device))) * 255
                    if self.dataset == 'RaFD':
                        x_fake = self.denorm(self.G(x_real, self.label2onehot(c_trg, self.c_dim).to(self.device))) * 255

                    x_fake = torch.round(x_fake).type(torch.uint8).cpu().numpy().transpose(0, 2, 3, 1)

                    for img, label in zip(x_fake, c_trg.numpy()):

                        # Dump images and label
                        if self.dataset == 'CelebA':
                            writer.add(img, {"attr": label})
                        elif self.dataset == 'RaFD':
                            writer.add(img, {"attr": np.array(label)})

                        f_count += 1
                        if f_count % 1000 == 0:
                            print('{} images created.'.format(f_count), end='\r')

            writer.close()
