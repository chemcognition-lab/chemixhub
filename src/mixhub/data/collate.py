import torch

from torch_geometric.data import Data, Batch
from mixhub.data.utils import UNK_TOKEN

from typing import List

def custom_collate(batch):
    """
    Custom collate function for CheMixHub
    """
    ids = torch.stack([item['ids'] for item in batch]).to(torch.int)
    fractions = torch.stack([item['fractions'] for item in batch])
    context = torch.stack([item['context'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])

    # Check type of first valid 'features' entry to determine how to collate
    if isinstance(batch[0]["features"], List):
        train_graphs = batch[0]["features"]
        features = Batch.from_data_list(train_graphs)
    else:
        features = torch.stack([item['features'] for item in batch])

    return {
        'ids': ids,
        'fractions': fractions,
        'context': context,
        'label': labels,
        'features': features,
    }
