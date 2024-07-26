import numpy as np
import torch

class FeatureExtractor:

    def __init__(self, model, patch_size, centre_size) -> None:
        self.model = model
        self.patch_size = patch_size
        self.centre_size = centre_size
        pass

    def get_mus(self, patch):
        hierarchy_level = self.model.n_layers
        n_channels = self.model.z_dims
        self.model.mode_pred = True
        self.model.eval()
        device = self.model.device
        with torch.no_grad():
            patch = patch.to(device=device, dtype=torch.float)
            patch = patch.reshape(1, 1, self.patch_size, self.patch_size)
            with torch.no_grad():
                sample = self.model(patch, patch, patch, model_layers=[i for i in range(hierarchy_level)])
                mu = sample["mu"]
        return mu

    def get_feature_maps(self, patch):
        """
        Return the feature maps for a patch using the model.
        """
        height, width = patch.shape
        mus = self.get_mus(patch)
        feature_map = [mus[i].flatten() for i in range(len(mus))]
        feature_map = np.array(torch.cat(feature_map,dim=-1).cpu().numpy())
        return feature_map


    def get_closest(self, train, test, sample_size):

        distances = np.linalg.norm(train - test, axis=1)
        closest_index = np.argmin(distances)
        if closest_index // sample_size == 0:
            return 0
        elif closest_index // sample_size == 1:
            return 1
        elif closest_index // sample_size == 2:
            return 2
        elif closest_index // sample_size == 3:
            return 3
        return None

        # # Initialize an array to hold the indices of the closest points in train for each point in test
        # closest_indices = np.zeros(test.shape[0], dtype=int)

        # # Initialize an array to hold the distances of the closest points
        # closest_distances = np.zeros(test.shape[0])

        # # For each point in test, find the closest point in train
        # for i in range(test.shape[0]):
        #     distances = np.linalg.norm(train - test[i], axis=1)
        #     closest_indices[i] = np.argmin(distances)
        #     closest_distances[i] = distances[closest_indices[i]]

        # # Get the closest points from array_1
        # closest_points = train[closest_indices]

        # return closest_indices, closest_points, closest_distances

