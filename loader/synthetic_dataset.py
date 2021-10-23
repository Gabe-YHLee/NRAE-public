from torch.utils import data
import numpy as np
import torch
import matplotlib.pyplot as plt
import math

class SyntheticData(data.Dataset):
    def __init__(self, split='training', type='sincurve', num_data=50, noise_level=0.0, graph=False, **kwargs):
        self.split = split
        self.data = self.get_data(type, num_data, noise_level)
        self.graph = graph
        if self.graph: 
            self.set_graph()
            
    def get_data(self, type, num_data, noise_level):
        '''
        For consistency (visualization purpose), we make the data lie in ``plt.xlim(-4, 4), plt.ylim(-1.5, 1.5)''
        '''
        if type == 'sincurve':
            x = np.linspace(-np.pi, np.pi, num_data)
            y = np.sin(x)
            data = torch.tensor([x, y], dtype=torch.float32).permute(1, 0) + noise_level*torch.randn(num_data, 2)
        elif type == 'swiss_role':
            num_samples = num_data
            r_max = float(3)
            r_min = float(0.1)
            r = [r_min + i*(r_max-r_min)/(num_samples-1) for i in range(num_samples)]
            theta_interval = 1/num_samples * 2 * math.pi
            theta = [i*theta_interval for i in range(num_samples)]
            data = torch.cat(
                [torch.tensor([r_*np.cos(theta_), 0.6*r_*np.sin(theta_)]).view(1, 2) for r_, theta_ in zip(r, theta)]
                , dim=0).to(torch.float32)
            data = data + noise_level*torch.randn(num_data, 2)
        return data
    
    def set_graph(self):
        data_temp = self.data.view(len(self.data), -1).clone()
        dist_mat = torch.cdist(data_temp, data_temp)
        dist_mat_indices = torch.topk(dist_mat, k=self.graph['num_nn'] + 1, dim=1, largest=False, sorted=True)
        self.dist_mat_indices = dist_mat_indices.indices[:, 1:]
    
    def visualize_data(self, training_data, test_data):
        f = plt.figure()
        plt.scatter(training_data[:, 0], training_data[:, 1], s=50, label='training data')
        plt.plot( test_data[:, 0], test_data[:, 1], linewidth=5 , c='k', label='data manifold')
        plt.title('Training Data and Ground Truth Data Manifold')
        plt.legend(loc='upper left')
        plt.xlim(-4, 4)
        plt.ylim(-1.5, 1.5)
        f.canvas.draw()
        f_arr = np.array(f.canvas.renderer._renderer)
        plt.close()
        return f_arr

    def visualize_graph(self, training_data, test_data, dist_mat_indices):
        f = plt.figure()
        plt.scatter(training_data[:, 0], training_data[:, 1], s=50, label='training data')
        plt.plot(test_data[:, 0], test_data[:, 1], linewidth=5 , c='k', label='data manifold')
        for i in range(len(training_data)):
            temp = dist_mat_indices[i]
            for j in range(len(temp)):
                line_arg = [(training_data[i][0], training_data[temp[j]][0]), (training_data[i][1], training_data[temp[j]][1]), '--']
                line_kwarg = {'c': 'g'}
                if (i == 0) and (j==0):
                    line_kwarg['label'] = 'neighborhood graph'
                plt.plot(*line_arg, **line_kwarg)
        plt.title('Neighborhood Graph')
        plt.legend(loc='upper left')
        plt.xlim(-4, 4)
        plt.ylim(-1.5, 1.5)
        f.canvas.draw()
        f_arr = np.array(f.canvas.renderer._renderer)
        plt.close()
        return f_arr
    
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
        
