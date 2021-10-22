from loader.synthetic_dataset import SyntheticData
from torch.utils import data

def get_dataloader(data_dict, **kwargs):
    dataset = get_dataset(data_dict)
    loader = data.DataLoader(
        dataset,
        batch_size=data_dict["batch_size"],
        shuffle=data_dict.get("shuffle", True)
    )
    return loader


def get_dataset(data_dict):
    name = data_dict["dataset"]
    if name == 'synthetic':
        dataset = SyntheticData(**data_dict)
    return dataset