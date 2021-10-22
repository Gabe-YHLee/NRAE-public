import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from matplotlib import collections  as mc

def get_kernel_function(kernel):
    if kernel['type'] == 'binary':
        def kernel_func(x_c, x_nn):
            '''
            x_c.size() = (bs, dim), 
            x_nn.size() = (bs, num_nn, dim)
            '''
            bs = x_nn.size(0)
            num_nn = x_nn.size(1)
            eps = 1.0e-12
            index = torch.norm(x_c.unsqueeze(1)-x_nn, dim=2) > eps
            output = torch.ones(bs, num_nn).to(x_c)
            output[~index] = kernel['lambda']
            return output # size: (bs, num_nn)
        return kernel_func
    
class AE(nn.Module):
    def __init__(self, encoder, decoder):
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

    def validation_step(self, x, **kwargs):
        recon = self(x)
        loss = ((recon - x) ** 2).view(len(x), -1).mean(dim=1).mean()
        return {"loss": loss.item()}

    def train_step(self, x, optimizer, **kwargs):
        optimizer.zero_grad()
        recon = self(x)
        loss = ((recon - x) ** 2).view(len(x), -1).mean(dim=1).mean()
        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}

    def visualize(self, epoch, training_loss, training_data, test_data, device):
        encoded = self.encoder(training_data.to(device))
        min_ = encoded.min().item()
        max_ = encoded.max().item()

        z_data = torch.tensor(
            np.linspace(min_,max_, 10000), 
            dtype=torch.float32).to(device).unsqueeze(1)
        gen_data = self.decoder(z_data).detach().cpu()
        recon_data = self.decoder(self.encoder(training_data.to(device))).detach().cpu()
        f = plt.figure()
        plt.plot(test_data[:, 0], test_data[:, 1], '--', linewidth=5 , c='k', label='data manifold')
        plt.plot(gen_data[:, 0], gen_data[:, 1], linewidth=5, c='tab:orange', label='learned manifold')
        for i in range(len(training_data)):
            line_arg = [
                (training_data[i][0], recon_data[i][0]), 
                (training_data[i][1], recon_data[i][1]), 
                '--']
            line_kwarg = {'c': 'r'}
            if i == 0:
                line_kwarg['label'] = 'point recon loss'
            plt.plot(*line_arg, **line_kwarg)
        plt.scatter(training_data[:, 0], training_data[:, 1], s=50, label='training data')
        plt.legend(loc='upper left')
        plt.title(f'epoch: {epoch}, training_loss: {training_loss:.4f}')
        plt.xlim(-4, 4)
        plt.ylim(-1.5, 1.5)
        f.canvas.draw()
        f_arr = np.array(f.canvas.renderer._renderer)
        plt.close()
        return f_arr

class NRAE(AE):
    def __init__(self, encoder, decoder, approx_order=1, kernel=None):
        super().__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder
        self.approx_order = approx_order
        self.kernel_func = get_kernel_function(kernel)
    
    def jacobian(self, z, dz, create_graph=True):
        batch_size = dz.size(0)
        num_nn = dz.size(1)
        z_dim = dz.size(2)

        v = dz.view(-1, z_dim)  # bs * num_nn , z_dim
        inputs = (
            z.unsqueeze(1).repeat(1, num_nn, 1).view(-1, z_dim)
        )  # bs * num_nn , z_dim

        jac = torch.autograd.functional.jvp(
            self.decoder, inputs, v=v, create_graph=create_graph
        )[1].view(
            batch_size, num_nn, -1
        )  # bs, num_nn, ::

        return jac        

    def jacobian_and_hessian(self, z, dz, create_graph=True):
        batch_size = dz.size(0)
        num_nn = dz.size(1)
        z_dim = dz.size(2)

        v = dz.view(-1, z_dim)  # bs * num_nn , z_dim
        inputs = (
            z.unsqueeze(1).repeat(1, num_nn, 1).view(-1, z_dim)
        )  # bs * num_nn , z_dim

        def jac_temp(inputs):
            jac = torch.autograd.functional.jvp(
                self.decoder, inputs, v=v, create_graph=create_graph
            )[1].view(
                batch_size, num_nn, -1
            )  # bs, num_nn, ::
            return jac

        temp = torch.autograd.functional.jvp(
            jac_temp, inputs, v=v, create_graph=create_graph
        )

        jac = temp[0].view(batch_size, num_nn, -1)
        hessian = temp[1].view(batch_size, num_nn, -1)  # bs, num_nears, ::
        return jac, hessian
        
    def neighborhood_recon(self, z_c, z_nn):
        recon = self.decoder(z_c)
        recon_x = recon.view(z_c.size(0), -1).unsqueeze(1)  # bs, 1, x_dim
        dz = z_nn - z_c.unsqueeze(1)  # bs, num_near, z_dim
        if self.approx_order == 1:
            Jdz = self.jacobian(z_c, dz)  # bs, num_near, x_dim
            n_recon = recon_x + Jdz
        elif self.approx_order == 2:
            Jdz, dzHdz = self.jacobian_and_hessian(z_c, dz)
            n_recon = recon_x + Jdz + 0.5*dzHdz
        return n_recon

    def train_step(self, x_c, x_nn, optimizer, **kwargs):
        optimizer.zero_grad()
        bs = x_nn.size(0)
        num_near = x_nn.size(1)

        z_c = self.encoder(x_c)
        z_dim = z_c.size(1)
        z_nn = self.encoder(
            x_nn.view(
                [-1] + list(x_nn.size()[2:]))
            ).view(bs, -1, z_dim)

        n_recon = self.neighborhood_recon(z_c, z_nn)
        
        n_loss = torch.norm(x_nn.view(bs, num_near, -1) - n_recon, dim=2) ** 2
        weights = self.kernel_func(x_c, x_nn)
        loss = (weights*n_loss).mean()
        loss.backward()
        optimizer.step()

        return {
            "loss": loss.item(),
        }

    def visualize(self, epoch, training_loss, training_data, test_data, device):
        encoded = self.encoder(training_data.to(device))
        min_ = encoded.min().item()
        max_ = encoded.max().item()

        z_data = torch.tensor(
            np.linspace(min_, max_, 10000), 
            dtype=torch.float32).to(device).unsqueeze(1)
        gen_data = self.decoder(z_data).detach().cpu()
        recon_data = self.decoder(self.encoder(training_data.to(device))).detach().cpu()
        f = plt.figure()
        plt.plot(test_data[:, 0], test_data[:, 1], '--', linewidth=5 , c='k', label='data manifold')
        plt.plot(gen_data[:, 0], gen_data[:, 1], linewidth=5, c='tab:orange', label='learned manifold')
        plt.scatter(training_data[:, 0], training_data[:, 1], s=50, label='training data')
        for i in range(len(training_data)):
            line_arg = [
                (training_data[i][0], recon_data[i][0]), 
                (training_data[i][1], recon_data[i][1]), 
                '--']
            line_kwarg = {'c': 'r', 'alpha': 0.1}
            if i == 0:
                line_kwarg['label'] = 'point recon loss'
            plt.plot(*line_arg, **line_kwarg) 
        
        plotted = 0
        plotted2 = 0
        for idx_of_interest in [int(0.2*len(training_data)), int(0.7*len(training_data))]:
            x_c = training_data[idx_of_interest:idx_of_interest+1]
            z_c = self.encoder(x_c.to(device))
            x_nn = training_data[self.dist_indices[idx_of_interest]]
            z_nn = self.encoder(x_nn.to(device))
            z_nn_recons = self.neighborhood_recon(z_c, z_nn.unsqueeze(0))[0].detach().cpu()
            for i in range(len(x_nn)):
                line_arg = [
                    (x_nn[i][0], z_nn_recons[i][0]), 
                    (x_nn[i][1], z_nn_recons[i][1]), 
                    '--']
                line_kwarg = {'c': 'tab:green'}
                if plotted == 0:
                    line_kwarg['label'] = 'neighborhood recon loss'
                    plotted = plotted + 1
                plt.plot(*line_arg, **line_kwarg) 
            z_min_ = z_nn.view(-1).min().item()
            z_max_ = z_nn.view(-1).max().item()    
            z_coordinates = torch.tensor(
                    np.linspace(z_min_ - 0.3, z_max_ + 0.3, 1000), 
                    dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(2)    
            n_recons = self.neighborhood_recon(z_c, z_coordinates)[0].detach().cpu()
            plt.scatter(x_c[:, 0], x_c[:, 1], c='tab:green')
            if plotted2 == 0:
                plt.plot(n_recons[:, 0], n_recons[:, 1], linewidth=5, c='tab:green', label='local approx. manifold')
                plotted2 = plotted2 + 1
            else:
                plt.plot(n_recons[:, 0], n_recons[:, 1], linewidth=5, c='tab:green')
            
        plt.legend(loc='upper left')
        plt.title(f'epoch: {epoch}, training_loss: {training_loss:.4f}')
        plt.xlim(-4, 4)
        plt.ylim(-1.5, 1.5)
        f.canvas.draw()
        f_arr = np.array(f.canvas.renderer._renderer)
        plt.close()
        return f_arr