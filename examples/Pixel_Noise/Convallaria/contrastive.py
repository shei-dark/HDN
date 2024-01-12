import torch
import torch.nn.functional as F
from torch.utils.data import Sampler, Dataset, DataLoader

class PositivePairSampler(Sampler):
    def __init__(self, labels):
        self.labels = labels
        self.label_indices = {label: torch.nonzero(labels == label).squeeze() for label in torch.unique(labels)}

    def __iter__(self):
        indices = []
        for label, indices_per_label in self.label_indices.items():
            # Ensure there are at least two instances of the label for a positive pair
            if len(indices_per_label) >= 2:
                indices.append(torch.randperm(len(indices_per_label)))

        return iter(indices)

    def __len__(self):
        return min(len(indices_per_label) // 2 for indices_per_label in self.label_indices.values())

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

def pixel_wise_contrastive_loss(z, sampler, temperature=1.0):
    all_indices = list(sampler)
    all_indices_flat = [idx.item() for indices_per_label in all_indices for idx in indices_per_label]

    z_flat = z.view(len(z), -1)
    z1 = z_flat[all_indices_flat]
    z2 = z_flat[all_indices_flat]

    # Compute cosine similarity between pixel representations
    similarities = F.cosine_similarity(z1, z2, dim=-1)

    # Apply temperature scaling
    similarities_scaled = similarities / temperature

    # Compute log probability for positive pairs
    log_prob_pos = F.log_softmax(similarities_scaled, dim=-1)

    # Extract log probability for positive pairs
    log_prob_pos = torch.cat([log_prob_pos[i, i].unsqueeze(0) for i in range(len(all_indices))])

    # Negative pairs (pixels from the same image but not in the positive pairs)
    log_prob_neg = torch.log(torch.sum(torch.exp(similarities_scaled), dim=-1))

    # Compute loss
    loss = - (log_prob_pos - log_prob_neg)

    return loss.mean()

# Example usage
labels = torch.randint(6, (100,))  # Assuming 6 different labels
sampler = PositivePairSampler(labels)

# Assuming you have a dataset with pixel representations (z) and corresponding labels
dataset = CustomDataset(torch.randn((100, 256)), labels)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler)

for batch in dataloader:
    z, labels_batch = batch
    loss = pixel_wise_contrastive_loss(z, sampler)
    print("Pixel-wise Contrastive Loss:", loss.item())
