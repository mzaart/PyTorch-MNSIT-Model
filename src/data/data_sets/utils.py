import torch


def get_all_rows(dataset, num_features):
    features = torch.zeros(len(dataset), num_features).float()
    labels = torch.zeros(len(dataset)).float()
    for i, row in enumerate(dataset):
        x, y = row['x'], row['y']
        features[i] = x
        labels[i] = y
    return {
        'x': features,
        'y': labels
    }
