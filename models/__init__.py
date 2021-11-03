import numpy as np

from models.ae import (
    AE,
    NRAE
)
from models.modules import (
    FC_vec,
    FC_image
)

def get_net(in_dim, out_dim, **kwargs):
    if kwargs["arch"] == "fc_vec":
        l_hidden = kwargs["l_hidden"]
        activation = kwargs["activation"]
        out_activation = kwargs["out_activation"]
        net = FC_vec(
            in_chan=in_dim,
            out_chan=out_dim,
            l_hidden=l_hidden,
            activation=activation,
            out_activation=out_activation,
        )
    elif kwargs["arch"] == "fc_image":
        l_hidden = kwargs["l_hidden"]
        activation = kwargs["activation"]
        out_activation = kwargs["out_activation"]
        net = FC_image(
            in_chan=in_dim,
            out_chan=out_dim,
            l_hidden=l_hidden,
            activation=activation,
            out_activation=out_activation,
        )
    return net

def get_ae(**model_cfg):
    x_dim = model_cfg["x_dim"]
    z_dim = model_cfg["z_dim"]
    if model_cfg["arch"] == "ae":
        encoder = get_net(in_dim=x_dim, out_dim=z_dim, **model_cfg["encoder"])
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **model_cfg["decoder"])
        ae = AE(encoder, decoder)
    elif model_cfg["arch"] == "nrael":
        encoder = get_net(in_dim=x_dim, out_dim=z_dim, **model_cfg["encoder"])
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **model_cfg["decoder"])
        ae = NRAE(encoder, decoder, approx_order=1, kernel=model_cfg["kernel"])
    elif model_cfg["arch"] == "nraeq":
        encoder = get_net(in_dim=x_dim, out_dim=z_dim, **model_cfg["encoder"])
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **model_cfg["decoder"])
        ae = NRAE(encoder, decoder, approx_order=2, kernel=model_cfg["kernel"])
    return ae

def get_model(model_cfg, *args, **kwargs):
    name = model_cfg["arch"]
    model = _get_model_instance(name)
    model = model(**model_cfg)
    return model

def _get_model_instance(name):
    try:
        return {
            "ae": get_ae,
            "nrael": get_ae,
            "nraeq": get_ae
        }[name]
    except:
        raise ("Model {} not available".format(name))