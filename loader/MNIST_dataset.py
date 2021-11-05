'''
rotated/shifted MNIST dataset
'''
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets.mnist import MNIST

class RotatedShiftedMNIST(MNIST):
    '''Select a single digit in MNIST then make a rotated or shifted dataset.
    '''
    def __init__(self,
        root,
        type='rotate',
        digit=0,
        graph=False,
        download=True,
        **kwargs):

        assert type in ["rotate", "shift"]

        super(RotatedShiftedMNIST, self).__init__(
            root,
            download=download,
        )

        data = self.data
        targets = self.targets

        data = (data.to(torch.float32) / 255).unsqueeze(1) # [data_size, 1, 28, 28]

        for idx, target in enumerate(targets):
            if target == digit:
                break
        data = data[idx]

        # Get dataset
        self.kwargs = kwargs
        self.data = self.get_data(type, data)
        print(f'MNIST dataset ready. {self.data.size()}')
        self.graph = graph
        if self.graph: 
            self.set_graph()

    def get_data(self, type, data):
        '''Generate rotated or shifted MNIST dataset
        '''
        data_list = []
        if type == 'rotate':
            num_rotate = self.kwargs.get('num_rotate', 100)
            rotate_range = self.kwargs.get('rotate_range', 180)
            for rot in [rotate_range*i/num_rotate for i in range(num_rotate)]:
                transformed_data = transforms.functional.affine(img=data, angle=rot, translate=[0, 0], scale=1., shear=0)
                data_list.append(transformed_data)
        elif type == 'shift':
            shift_range = self.kwargs.get('shift_range', 10)
            shift_range = [int(-abs(shift_range)), int(abs(shift_range))]
            for sh in range(shift_range[0], shift_range[1]):
                transformed_data = transforms.functional.affine(img=data, angle=0, translate=[sh, 0], scale=0.7, shear=0)
                data_list.append(transformed_data)
        data_list = torch.stack(data_list, dim=0)
        return data_list
    
    def set_graph(self):
        data_temp = self.data.view(len(self.data), -1).clone()
        dist_mat = torch.cdist(data_temp, data_temp)
        dist_mat_indices = torch.topk(dist_mat, k=self.graph['num_nn'] + 1, dim=1, largest=False, sorted=True)
        self.dist_mat_indices = dist_mat_indices.indices[:, 1:]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.graph:
            bs_nn = self.graph['bs_nn']
            if self.graph['include_center']:
                x_c = self.data[idx]
                x_nn = self.data[
                    self.dist_mat_indices[
                        idx, 
                        np.random.choice(range(self.graph['num_nn']), bs_nn-1, replace=self.graph['replace'])
                    ]
                ]
                return x_c, torch.cat([x_c.unsqueeze(0), x_nn], dim=0)
            else:
                x_c = self.data[idx]
                x = self.data[
                    self.dist_mat_indices[
                        idx, 
                        np.random.choice(range(self.graph['num_nn']), bs_nn, replace=self.graph['replace'])
                    ]
                ]
                return x_c, x
        else:
            x = self.data[idx]
            return x