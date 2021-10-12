import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision.utils import make_grid

class AE(nn.Module):
    def __init__(self, encoder, decoder):
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z = self.encode(x)
        recon = self.decoder(z)
        return recon

    def encode(self, x):
        z = self.encoder(x)
        return z

    def encoder_mean(self, x):
        return self.encoder(x)

    def decoder_mean(self, z):
        return self.decoder(z)

    def predict(self, x):
        recon = self(x)
        predict = ((recon - x) ** 2).view(len(x), -1).mean(dim=1)
        return predict

    def validation_step(self, x, **kwargs):
        recon = self(x)
        if hasattr(self.decoder, "error"):
            predict = self.decoder.error(x, recon)
        else:
            predict = ((recon - x) ** 2).view(len(x), -1).mean(dim=1)
        loss = predict.mean()

        if kwargs.get("show_image", True):
            x_img = make_grid(x.detach().cpu(), nrow=10, range=(0, 1))
            recon_img = make_grid(recon.detach().cpu(), nrow=10, range=(0, 1))
        else:
            x_img, recon_img = None, None
        return {
            "loss": loss.item(),
            "val_reconstruction#": recon,
            "val_input#": x,
            "ximag@": x_img,
            "recon@": recon_img,
        }

    def train_step(self, x, optimizer, clip_grad=None, **kwargs):
        optimizer.zero_grad()
        recon = self(x)
        recon_error = self.predict(x)
        loss = recon_error.mean()
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_grad)
        optimizer.step()
        return {"loss": loss.item()}


class NRAE_L(AE):
    def __init__(self, encoder, decoder, lamb=0.001):
        super().__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder
        self.lamb = lamb

    def validation_step(self, x, optimizer=None, y=None):
        z = self.encoder(x)
        recon = self.decoder(z)
        recon_loss = ((x - recon)) ** 2
        loss = recon_loss.mean()

        if True:
            x_img = make_grid(x.detach().cpu(), nrow=10, range=(0, 1))
            recon_img = make_grid(recon.detach().cpu(), nrow=10, range=(0, 1))
        else:
            x_img, recon_img = None, None

        return {
            "loss": loss.item(),
            "recon_loss": recon_loss.mean().item(),
            "val_input#": x,
            "val_reconstruction#": recon,
            "ximg@": x_img,
            "reconimg@": recon_img,
        }

    def train_step(self, x, x_nn, optimizer=None, y=None):
        optimizer.zero_grad()

        z = self.encoder(x)
        x_size = x.size()
        bs = x_size[0]
        z_dim = z.size(1)
        num_near = x_nn.size(1)

        if len(x_size)==4:
            z_nn = self.encoder(x_nn.view(-1, x_size[1], x_size[2], x_size[3])).view(bs, -1, z_dim)
        elif len(x_size)==2:
            z_nn = self.encoder(x_nn.view(-1, x_size[1])).view(bs, -1, z_dim)
        
        recon = self.decoder(z)
        recon_loss = torch.mean(((x - recon) ** 2).view(bs, -1), dim=1)
        
        dz = z_nn - z.unsqueeze(1)  # bs, num_near, z_dim
        Jdz = self.jacobian(z, dz)  # bs, num_near, x_dim
        recon_x = recon.view(bs, -1).unsqueeze(1)  # bs, 1, x_dim
        n_recon = recon_x + Jdz
        n_loss = torch.mean(
            torch.norm(x_nn.view(bs, num_near, -1) - n_recon, dim=2) ** 2, dim=1
        )

        loss = (recon_loss + self.lamb * n_loss).mean()
        loss.backward()
        optimizer.step()

        return {
            "loss": loss.item(),
            "recon_loss_": recon_loss.mean().item(),
            "lqr_loss": n_loss.mean().item(),
            "train_input": x,
            "train_reconstruction": recon,
        }

    def jacobian(self, z, dz, create_graph=True):
        batch_size = dz.size(0)
        num_near = dz.size(1)
        z_dim = dz.size(2)

        v = dz.view(-1, z_dim)  # bs * num_nears , z_dim
        inputs = (
            z.unsqueeze(1).repeat(1, num_near, 1).view(-1, z_dim)
        )  # bs * num_nears , z_dim

        jac = torch.autograd.functional.jvp(
            self.decoder, inputs, v=v, create_graph=create_graph
        )[1].view(
            batch_size, num_near, -1
        )  # bs, num_nears, ::

        return jac
    

class NRAE_Q(AE):
    def __init__(self, encoder, decoder, lamb=0.001):
        super().__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder
        self.lamb = lamb

    def validation_step(self, x, optimizer=None, y=None):
        z = self.encoder(x)
        recon = self.decoder(z)
        recon_loss = ((x - recon)) ** 2
        loss = recon_loss.mean()

        if True:
            x_img = make_grid(x.detach().cpu(), nrow=10, range=(0, 1))
            recon_img = make_grid(recon.detach().cpu(), nrow=10, range=(0, 1))
        else:
            x_img, recon_img = None, None

        return {
            "loss": loss.item(),
            "recon_loss": recon_loss.mean().item(),
            "val_input#": x,
            "val_reconstruction#": recon,
            "ximg@": x_img,
            "reconimg@": recon_img,
        }
    def train_step(self, x, x_nn, optimizer=None, y=None):
        optimizer.zero_grad()

        z = self.encoder(x)
        x_size = x.size()
        bs = x_size[0]
        z_dim = z.size(1)
        num_near = x_nn.size(1)
        if len(x_size)==4:
            z_nn = self.encoder(x_nn.view(-1, x_size[1], x_size[2], x_size[3])).view(bs, -1, z_dim)
        elif len(x_size)==2:
            z_nn = self.encoder(x_nn.view(-1, x_size[1])).view(bs, -1, z_dim)

        recon = self.decoder(z)
        recon_loss = torch.mean(((x - recon) ** 2).view(bs, -1), dim=1)

        dz = z_nn - z.unsqueeze(1)  # bs, num_near, z_dim
        Jdz, dzHdz = self.jacobian_and_hessian_jvp_ver_for_decoder(z, dz)  # bs, num_near, x_dim
        recon_x = recon.view(bs, -1).unsqueeze(1)  # bs, 1, x_dim

        n_recon = recon_x + Jdz + 1 / 2 * dzHdz
        n_loss = torch.mean(torch.norm(x_nn.view(bs, num_near, -1) - n_recon, dim=2) ** 2, dim=1)

        loss = (recon_loss + self.lamb * n_loss).mean()
        loss.backward()
        optimizer.step()

        return {
            "loss": loss.item(),
            "recon_loss_": recon_loss.mean().item(),
            "lqr_loss": n_loss.mean().item(),
            "train_input": x,
            "train_reconstruction": recon,
        }

    def jacobian_and_hessian_jvp_ver_for_decoder(self, z, dz, create_graph=True):
        batch_size = dz.size(0)
        num_near = dz.size(1)
        z_dim = dz.size(2)

        v = dz.view(-1, z_dim)  # bs * num_nears , z_dim
        inputs = (
            z.unsqueeze(1).repeat(1, num_near, 1).view(-1, z_dim)
        )  # bs * num_nears , z_dim

        def jac_temp(inputs):
            jac = torch.autograd.functional.jvp(
                self.decoder, inputs, v=v, create_graph=create_graph
            )[1].view(
                batch_size, num_near, -1
            )  # bs, num_nears, ::
            return jac

        temp = torch.autograd.functional.jvp(
            jac_temp, inputs, v=v, create_graph=create_graph
        )

        jac = temp[0].view(batch_size, num_near, -1)
        hessian = temp[1].view(batch_size, num_near, -1)  # bs, num_nears, ::

        return jac, hessian