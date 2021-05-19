import os.path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision

import urllib.request
from PIL import Image
from PIL import ExifTags
import math

import pandas as pd
import numpy as np

from tqdm import tqdm_notebook as tqdm
import glob

scene_name = "Madrid_Metropolis"
dataset_path = "d:/Kutatas/PoseGraphRANSAC/git/data/Madrid_Metropolis/images/"

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def load_imglist(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
        fnames = [x.split(' ')[0].strip() for x in lines]
    return fnames

class RetrievalDataset(torch.utils.data.Dataset):
    """Retrieval dataset."""
    def __init__(self, imglist_fname, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = load_imglist(imglist_fname)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame[idx])
        image = np.array(Image.open(img_name).convert('RGB'))

        if self.transform:
            image = self.transform(image)

        return image


class ImageRanker():
    def __init__(self, model, transforms=transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])):
        self.model = model
        self.dev = torch.device('cpu')
        self.model.eval()
        self.model = self.model.to(self.dev)
        self.transforms = transforms
        self.dim = 0
        self.n_imgs = 0
        return

    def process_db_images(self, filelist_fname, root_dir='.', on_gpu=False,
                          db_save_fname='demo_db.pth', do_diffusion=False):
        if on_gpu:
            try:
                on_gpu = on_gpu and torch.cuda.is_available()
                self.dev = torch.device('cuda:0')
                self.model = self.model.to(self.dev)
            except:
                print('GPU is not available')
        self.dataset = RetrievalDataset(filelist_fname, root_dir, transform=self.transforms)
        self.n_imgs = len(self.dataset)
        bs = 1
        params = {'batch_size': bs,
                  'shuffle': False,
                  'num_workers': 0}
        self.dataloader = torch.utils.data.DataLoader(self.dataset, **params)
        self.features = None
        if os.path.isfile(db_save_fname):
            print('Loading features from disk')
            self.load_db(db_save_fname)
            print('Done')
            return
        print('Describing images...')
        for idx, data in tqdm(enumerate(self.dataloader), total=self.n_imgs // bs):
            if idx % 100 == 0:
                print(f"Describing image [{idx} / {self.n_imgs}]")
            with torch.no_grad():
                current_feats = F.normalize(self.model(data.to(self.dev)), dim=1, p=2)
                st = idx * bs
                fin = st + len(current_feats)
                if idx == 0:
                    self.dim = current_feats.size(1)
                    self.features = torch.zeros(self.n_imgs, self.dim, dtype=torch.float, device=self.dev)
                self.features[st:fin] = current_feats
                st = fin
        if do_diffusion:
            self.K = 100
            with torch.no_grad():
                W = torch.mm(self.features, self.features.t()).pow(3).clamp_(min=0)
                W = topK_W(W.detach().cpu().numpy(), 100)
            W = topK_W(W, self.K)
            self.W = normalize_connection_graph(W)
        else:
            with torch.no_grad():
                self.W = torch.mm(self.features, self.features.t()).pow(3).clamp_(min=0).detach().cpu().numpy()
        try:
            self.save_db(db_save_fname)
        except:
            print('Failed to save features to {}'.format(db_save_fname))
        print('Done')
        return

    def save_db(self, fname):
        torch.save({'features': self.features.cpu(), 'connect_graph': self.W}, fname)
        return

    def load_db(self, fname):
        load = torch.load(fname, map_location=torch.device('cpu'))
        self.features = load['features']
        self.n_imgs = self.features.size(0)
        self.dim = self.features.size(1)
        self.W = load['connect_graph']
        return

    def get_similar(self, img, num_nn=10, use_diffusion=False):
        transformed_img = self.transforms(img).to(self.dev).unsqueeze(0)
        with torch.no_grad():
            descriptor = F.normalize(self.model(transformed_img), dim=1, p=2)
            dists = torch.cdist(descriptor, self.features)
            out_dists, out_idxs = torch.topk(dists, k=min(self.n_imgs, num_nn), dim=1, largest=False)
            out_dists = out_dists.cpu().detach().numpy().flatten()
            out_idxs = out_idxs.cpu().detach().numpy().flatten()
            if use_diffusion:
                QUERYKNN = 10
                R = min(self.n_imgs, 100)
                alpha = 0.9
                qsim = torch.relu((1.0 - dists / 2.).pow(3)).detach().numpy()
                sortidxs = np.argsort(-qsim, axis=1)
                for i in range(len(qsim)):
                    qsim[i, sortidxs[i, QUERYKNN:]] = 0
                print(qsim.shape)
                ranks = cg_diffusion(qsim, self.W, alpha)[:num_nn, 0]
                # print (ranks.shape)
                out_idxs = ranks
        return {'dists': out_dists,
                'idxs': out_idxs,
                'paths': [str(self.dataset.landmarks_frame.iloc[i, 0]) for i in out_idxs],
                }


# Took from https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/master/cirtorch/networks/imageretrievalnet.py
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


def l2n(x, eps=1e-6):
    return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)


def powerlaw(x, eps=1e-6):
    x = x + self.eps
    return x.abs().sqrt().mul(x.sign())


class L2N(nn.Module):

    def __init__(self, eps=1e-6):
        super(L2N, self).__init__()
        self.eps = eps

    def forward(self, x):
        return l2n(x, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'eps=' + str(self.eps) + ')'


class PowerLaw(nn.Module):

    def __init__(self, eps=1e-6):
        super(PowerLaw, self).__init__()
        self.eps = eps

    def forward(self, x):
        return LF.powerlaw(x, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'eps=' + str(self.eps) + ')'


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' \
               + 'eps=' + str(self.eps) + ')'


OUTPUT_DIM = {
    'alexnet': 256,
    'vgg11': 512,
    'vgg13': 512,
    'vgg16': 512,
    'vgg19': 512,
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'resnet152': 2048,
    'densenet121': 1024,
    'densenet169': 1664,
    'densenet201': 1920,
    'densenet161': 2208,  # largest densenet
    'squeezenet1_0': 512,
    'squeezenet1_1': 512,
}

class ImageRetrievalNet(nn.Module):

    def __init__(self, features, lwhiten, pool, whiten, meta):
        super(ImageRetrievalNet, self).__init__()
        self.features = nn.Sequential(*features)
        self.lwhiten = lwhiten
        self.pool = pool
        self.whiten = whiten
        self.norm = L2N()
        self.meta = meta

    def forward(self, x):
        # x -> features
        o = self.features(x)

        # TODO: properly test (with pre-l2norm and/or post-l2norm)
        # if lwhiten exist: features -> local whiten
        if self.lwhiten is not None:
            # o = self.norm(o)
            s = o.size()
            o = o.permute(0, 2, 3, 1).contiguous().view(-1, s[1])
            o = self.lwhiten(o)
            o = o.view(s[0], s[2], s[3], self.lwhiten.out_features).permute(0, 3, 1, 2)
            # o = self.norm(o)

        # features -> pool -> norm
        o = self.norm(self.pool(o)).squeeze(-1).squeeze(-1)

        # if whiten exist: pooled features -> whiten -> norm
        if self.whiten is not None:
            o = self.norm(self.whiten(o))

        # permute so that it is Dx1 column vector per image (DxN if many images)
        return o  # .permute(1,0)

    def __repr__(self):
        tmpstr = super(ImageRetrievalNet, self).__repr__()[:-1]
        tmpstr += self.meta_repr()
        tmpstr = tmpstr + ')'
        return tmpstr

    def meta_repr(self):
        tmpstr = '  (' + 'meta' + '): dict( \n'  # + self.meta.__repr__() + '\n'
        tmpstr += '     architecture: {}\n'.format(self.meta['architecture'])
        tmpstr += '     local_whitening: {}\n'.format(self.meta['local_whitening'])
        tmpstr += '     pooling: {}\n'.format(self.meta['pooling'])
        tmpstr += '     regional: {}\n'.format(self.meta['regional'])
        tmpstr += '     whitening: {}\n'.format(self.meta['whitening'])
        tmpstr += '     outputdim: {}\n'.format(self.meta['outputdim'])
        tmpstr += '     mean: {}\n'.format(self.meta['mean'])
        tmpstr += '     std: {}\n'.format(self.meta['std'])
        tmpstr = tmpstr + '  )\n'
        return tmpstr


def init_network(params):
    # parse params with default values
    architecture = params.get('architecture', 'resnet50')
    local_whitening = params.get('local_whitening', False)
    pooling = params.get('pooling', 'gem')
    regional = params.get('regional', False)
    whitening = params.get('whitening', False)
    mean = params.get('mean', [0.485, 0.456, 0.406])
    std = params.get('std', [0.229, 0.224, 0.225])
    pretrained = params.get('pretrained', True)

    # get output dimensionality size
    dim = OUTPUT_DIM[architecture]

    # loading network from torchvision
    if pretrained:
        if architecture not in FEATURES:
            # initialize with network pretrained on imagenet in pytorch
            net_in = getattr(torchvision.models, architecture)(pretrained=True)
        else:
            # initialize with random weights, later on we will fill features with custom pretrained network
            net_in = getattr(torchvision.models, architecture)(pretrained=False)
    else:
        # initialize with random weights
        net_in = getattr(torchvision.models, architecture)(pretrained=False)

    # initialize features
    # take only convolutions for features,
    # always ends with ReLU to make last activations non-negative
    if architecture.startswith('alexnet'):
        features = list(net_in.features.children())[:-1]
    elif architecture.startswith('vgg'):
        features = list(net_in.features.children())[:-1]
    elif architecture.startswith('resnet'):
        features = list(net_in.children())[:-2]
    elif architecture.startswith('densenet'):
        features = list(net_in.features.children())
        features.append(nn.ReLU(inplace=True))
    elif architecture.startswith('squeezenet'):
        features = list(net_in.features.children())
    else:
        raise ValueError('Unsupported or unknown architecture: {}!'.format(architecture))

    # initialize local whitening
    if local_whitening:
        lwhiten = nn.Linear(dim, dim, bias=True)
        # TODO: lwhiten with possible dimensionality reduce

        if pretrained:
            lw = architecture
            if lw in L_WHITENING:
                print(">> {}: for '{}' custom computed local whitening '{}' is used"
                      .format(os.path.basename(__file__), lw, os.path.basename(L_WHITENING[lw])))
                whiten_dir = os.path.join(get_data_root(), 'whiten')
                lwhiten.load_state_dict(model_zoo.load_url(L_WHITENING[lw], model_dir=whiten_dir))
            else:
                print(">> {}: for '{}' there is no local whitening computed, random weights are used"
                      .format(os.path.basename(__file__), lw))

    else:
        lwhiten = None

    # initialize pooling
    if pooling == 'gemmp':
        pool = POOLING[pooling](mp=dim)
    else:
        pool = POOLING[pooling]()

    # initialize regional pooling
    if regional:
        rpool = pool
        rwhiten = nn.Linear(dim, dim, bias=True)
        # TODO: rwhiten with possible dimensionality reduce

        if pretrained:
            rw = '{}-{}-r'.format(architecture, pooling)
            if rw in R_WHITENING:
                print(">> {}: for '{}' custom computed regional whitening '{}' is used"
                      .format(os.path.basename(__file__), rw, os.path.basename(R_WHITENING[rw])))
                whiten_dir = os.path.join(get_data_root(), 'whiten')
                rwhiten.load_state_dict(model_zoo.load_url(R_WHITENING[rw], model_dir=whiten_dir))
            else:
                print(">> {}: for '{}' there is no regional whitening computed, random weights are used"
                      .format(os.path.basename(__file__), rw))

        pool = Rpool(rpool, rwhiten)

    # initialize whitening
    if whitening:
        whiten = nn.Linear(dim, dim, bias=True)
        # TODO: whiten with possible dimensionality reduce

        if pretrained:
            w = architecture
            if local_whitening:
                w += '-lw'
            w += '-' + pooling
            if regional:
                w += '-r'
            if w in WHITENING:
                print(">> {}: for '{}' custom computed whitening '{}' is used"
                      .format(os.path.basename(__file__), w, os.path.basename(WHITENING[w])))
                whiten_dir = os.path.join(get_data_root(), 'whiten')
                whiten.load_state_dict(model_zoo.load_url(WHITENING[w], model_dir=whiten_dir))
            else:
                print(">> {}: for '{}' there is no whitening computed, random weights are used"
                      .format(os.path.basename(__file__), w))
    else:
        whiten = None

    # create meta information to be stored in the network
    meta = {
        'architecture': architecture,
        'local_whitening': local_whitening,
        'pooling': pooling,
        'regional': regional,
        'whitening': whitening,
        'mean': mean,
        'std': std,
        'outputdim': dim,
    }

    # create a generic image retrieval network
    net = ImageRetrievalNet(features, lwhiten, pool, whiten, meta)

    # initialize features with custom pretrained network if needed
    if pretrained and architecture in FEATURES:
        print(">> {}: for '{}' custom pretrained features '{}' are used"
              .format(os.path.basename(__file__), architecture, os.path.basename(FEATURES[architecture])))
        model_dir = os.path.join(get_data_root(), 'networks')
        net.features.load_state_dict(model_zoo.load_url(FEATURES[architecture], model_dir=model_dir))

    return net

POOLING = {
    'gem': GeM,
}

if not os.path.isfile('gl18-tl-resnet50-gem-w-83fdc30.pth'):
    print("Downloaded Resnet50 model")
    url = "http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet50-gem-w-83fdc30.pth"
    urllib.request.urlretrieve(url, 'gl18-tl-resnet50-gem-w-83fdc30.pth')

net_params = {}
net_params['architecture'] = 'resnet50'
net_params['pooling'] = 'gem'
net_params['local_whitening'] = False
net_params['regional'] = False
net_params['whitening'] =True
net_params['mean'] =[0.485, 0.456, 0.406]
net_params['std'] = [0.229, 0.224, 0.225]
net_params['pretrained'] = False
# network initialization
net = init_network(net_params)

sd = torch.load('gl18-tl-resnet50-gem-w-83fdc30.pth', map_location=torch.device('cpu'))
net.load_state_dict(sd['state_dict'])
net.eval()

imglist_fname = os.path.join(dataset_path, 'list.txt')
extensions = ("*.png", "*.jpg", "*.jpeg",)

if not os.path.exists(imglist_fname):
    print(f"Creating image list file '{imglist_fname}'.")

    f = open(imglist_fname, "w")
    for ext in extensions:
        for filename in glob.iglob(dataset_path + ext, recursive=False):
            f.write(filename + "\n")
    f.close()

IR = ImageRanker(net)
out_fname = dataset_path + "/" + scene_name + '_resnet50_similarity.txt'
out_db_fname = dataset_path + "/" + scene_name + '.pth'
IR.process_db_images(imglist_fname, dataset_path, do_diffusion=False, on_gpu=True, db_save_fname=out_db_fname)
IR.W = torch.mm(IR.features, IR.features.t()).clamp_(min=0).detach().cpu().numpy()
np.savetxt(out_fname, IR.W, fmt='%1.3f')
