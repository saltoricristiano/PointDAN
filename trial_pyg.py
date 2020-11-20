import torch
import torch.utils.data as data
import os
import sys
import h5py
import numpy as np
import glob
import random
import torchvision
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import torch_geometric.nn as nn
import torch
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm


def load_dir(data_dir, name='train_files.txt'):
    with open(os.path.join(data_dir,name),'r') as f:
        lines = f.readlines()
    return [os.path.join(data_dir, line.rstrip().split('/')[-1]) for line in lines]


def get_info(shapes_dir, isView=False):
    names_dict = {}
    if isView:
        for shape_dir in shapes_dir:
            name = '_'.join(os.path.split(shape_dir)[1].split('.')[0].split('_')[:-1])
            if name in names_dict:
                names_dict[name].append(shape_dir)
            else:
                names_dict[name] = [shape_dir]
    else:
        for shape_dir in shapes_dir:
            name = os.path.split(shape_dir)[1].split('.')[0]
            names_dict[name] = shape_dir

    return names_dict


def normal_pc(pc):
    """
    normalize point cloud in range L
    :param pc: type list
    :return: type list
    """
    pc_mean = pc.mean(axis=0)
    pc = pc - pc_mean
    pc_L_max = np.max(np.sqrt(np.sum(abs(pc ** 2), axis=-1)))
    pc = pc/pc_L_max
    return pc


class Rotate(object):
    def __call__(self, data):
        """
        Randomly rotate the point clouds to augment the dataset
        rotation is per shape based along up direction
        :param pc: B X N X 3 array, original batch of point clouds
        :return: BxNx3 array, rotated batch of point clouds
        """
        # rotated_data = np.zeros(pc.shape, dtype=np.float32)
        pc = data.pos
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        # rotation_matrix = np.array([[cosval, 0, sinval],
        #                             [0, 1, 0],
        #                             [-sinval, 0, cosval]])
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, cosval, -sinval],
                                    [0, sinval, cosval]])
        # rotation_matrix = np.array([[cosval, -sinval, 0],
        #                             [sinval, cosval, 0],
        #                             [0, 0, 1]])
        rotated_data = np.dot(pc.reshape((-1, 3)), rotation_matrix)
        data.pos = torch.from_numpy(rotated_data)

        return data


class Jitter_PCL(object):
    def __call__(self, data, sigma=0.01, clip=0.05):
        """
        Randomly jitter points. jittering is per point.
        :param pc: B X N X 3 array, original batch of point clouds
        :param sigma:
        :param clip:
        :return:
        """
        pc = data.pos
        jittered_data = torch.from_numpy(np.clip(sigma * np.random.randn(*pc.shape), -1 * clip, clip))
        jittered_data += pc
        data.pos = jittered_data
        data.edge_index = nn.radius_graph(x=torch.tensor(pc), r=0.05)
        return data


class Shapenet_data(InMemoryDataset):
    def __init__(self, data_root, status='train', aug=False, pc_input_num=1024, data_type='*.npy'):
        self.status = status
        self.transform = torchvision.transforms.Compose([Rotate()]) if aug else None
        self.pc_input_num = pc_input_num
        self.data_type = data_type
        self.data_root = data_root
        super(Shapenet_data, self).__init__('dataset/processed/shapenet', self.transform, None)
        self.data, self.slices = torch.load(os.path.join(self.processed_dir, self.status, 'process.pt'))

    def prepr_item(self, idx):
        lbl = self.lbl_list[idx]
        if self.data_type == '*.pts':
            pc = np.array([[float(value) for value in xyz.split(' ')]
                           for xyz in open(self.pc_list[idx], 'r') if len(xyz.split(' ')) == 3])[:self.pc_input_num, :]
        elif self.data_type == '*.npy':
            pc = np.load(self.pc_list[idx])[:self.pc_input_num].astype(np.float32)
        pc = normal_pc(pc)
        pad_pc = np.zeros(shape=(self.pc_input_num-pc.shape[0], 3), dtype=float)
        pc = np.concatenate((pc, pad_pc), axis=0)

        edge_idx = nn.radius_graph(x=torch.tensor(pc), r=0.05).long()
        d = Data(pos=torch.tensor(pc), edge_index=edge_idx, y=torch.tensor([float(lbl)]))
        d.num_nodes=self.pc_input_num

        return d

    @property
    def raw_file_names(self):
        return self.pc_list

    @property
    def processed_file_names(self):
        return [os.path.join(self.status, 'process.pt')]

    def process(self):

        self.pc_list = []
        self.lbl_list = []
        

        categorys = glob.glob(os.path.join(self.data_root, '*'))
        categorys = [c.split(os.path.sep)[-1] for c in categorys]
        categorys = sorted(categorys)

        if self.status == 'train':
            self.pts_list = glob.glob(os.path.join(self.data_root, '*', 'train', self.data_type))
        elif self.status == 'test':
            self.pts_list = glob.glob(os.path.join(self.data_root, '*', 'test', self.data_type))
        else:
            self.pts_list = glob.glob(os.path.join(self.data_root, '*', 'validation', self.data_type))
        # names_dict = get_info(pts_list, isView=False)

        for _dir in self.pts_list:
            self.pc_list.append(_dir)
            self.lbl_list.append(categorys.index(_dir.split('/')[-3]))

        # Read data into huge `Data` list.
        data_list = []
        for i in tqdm(range(len(self.pc_list))):
            data_list.append(self.prepr_item(i))

        data, slices = self.collate(data_list)
        os.makedirs(os.path.join(self.processed_dir, self.status), exist_ok=True)
        torch.save((data, slices), os.path.join(self.processed_dir, self.status, 'process.pt'))


if __name__ == "__main__":

    trainset = Shapenet_data(data_root='./dataset/PointDA_data/shapenet', status='train', aug=True, data_type='*.npy')
    train_loader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)

    testset = Shapenet_data(data_root='./dataset/PointDA_data/shapenet', status='test', aug=False, data_type='*.npy')
    test_loader = DataLoader(testset, batch_size=32, shuffle=True, num_workers=4)    

