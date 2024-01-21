import torch
import torch.nn.functional as F

def metric(z1, z2):
    return F.cosine_similarity(z1, z2, dim=-1)

def nc2(n):
    return n*(n-1)/2


def contrastive_learning_loss(z, labels):
    """
    z: (batch_size, C, H, W)
    labels: (batch_size)
    """
    unique_labels = torch.unique(labels)
    for label in unique_labels:
        positive_index_mask = labels == label
        negative_index_mask = labels != label

        positive_z = z[positive_index_mask]
        negative_z = z[negative_index_mask]
        
        positive_loss = 0
        negative_loss = 0 
        for i in range(len(positive_z)):
            for j in range(len(positive_z)):
                if i != j:
                    positive_loss += metric(positive_z[i], positive_z[j])
        

        for i in range(len(positive_z)):
            for j in range(len(negative_z)):
                negative_loss += metric(positive_z[i], negative_z[j])

    # give a weight to balance it. 
        positive_loss = positive_loss/ nc2(len(positive_z))
        negative_loss = negative_loss/ (len(positive_z) * len(negative_z))

    return positive_loss - negative_loss
