import os
from glob import glob
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from PIL import Image
from torchvision import transforms as T
from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1, device=0):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        torch.cuda.set_device(device)
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)
        if (i + 1) * batch_size % 100 == 0:
            print('Computed {} embeddings.'.format((i + 1) * batch_size))

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

class FakeImageDataset(torch.utils.data.Dataset):
    def __init__(self, result_dir):
        self.result_dir = result_dir
        self.img_dir = os.path.join(self.result_dir, 'fake_all')
        self.img_list = glob(os.path.join(self.img_dir, '*.jpg'))
        self.transform = T.Compose([T.ToTensor(),
                                    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        filepath = self.img_list[index]
        return self.transform(Image.open(filepath))

    def __len__(self):
        return len(self.img_list)

if __name__ == '__main__':
    result_dir = 'stargan_celeba/results_5selected_sensible_156labelled_allunlabelled_sorted_cls_conv'
    fake_dataset = FakeImageDataset(result_dir)

    print("Calculating Inception Score...")
    m, std = inception_score(fake_dataset, cuda=True, batch_size=32, resize=True, splits=10)
    print(m, std)

    file_path = os.path.join(result_dir, 'inception_score.txt')
    with open(file_path, 'w') as f:
        f.write('Inception score: {:.4f}, standard deviation: {:.4f}'.format(m, std))
