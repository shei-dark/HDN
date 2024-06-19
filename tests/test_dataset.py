import pytest
from lib.dataloader import CustomDataset
import numpy as np


@pytest.fixture
def mock_data():
    images = [np.random.randint(0, 255, (128, 128), dtype=np.uint8) for _ in range(5)]
    labels = [np.random.randint(0, 3, (128, 128), dtype=np.uint8) for _ in range(5)]  # Using 3 classes for example
    return images, labels


def test_init(mock_data):
    images, labels = mock_data
    dataset = CustomDataset(images, labels, patch_size=64, mask_size=4)
    assert dataset.patch_size == 64
    assert dataset.mask_size == 4
    assert isinstance(dataset.patches_by_label, dict)

def test_extract_valid_patches(mock_data):
    images, labels = mock_data
    dataset = CustomDataset(images, labels, patch_size=64, mask_size=4)
    patches_by_label = dataset._extract_valid_patches(images, labels)
    
    assert isinstance(patches_by_label, dict)
    for label, patches in patches_by_label.items():
        assert isinstance(label, int)
        assert isinstance(patches, list)
        for patch in patches:
            assert patch.shape == (1, 64, 64)

def test_getitem(mock_data):
    images, labels = mock_data
    dataset = CustomDataset(images, labels, patch_size=64, mask_size=4)
    
    # Ensure the dataset has valid patches for at least one label
    if dataset.patches_by_label:
        for label in dataset.patches_by_label.keys():
            patches = dataset.__getitem__(label, num_patches=2)
            assert len(patches) == 2
            for patch in patches:
                assert patch.shape == (1, 64, 64)
            break  # Test only one label

def test_getitem_exceeds(mock_data):
    images, labels = mock_data
    dataset = CustomDataset(images, labels, patch_size=64, mask_size=4)
    
    # Ensure the dataset has valid patches for at least one label
    if dataset.patches_by_label:
        for label in dataset.patches_by_label.keys():
            total_patches = len(dataset.patches_by_label[label])
            patches = dataset.__getitem__(label, num_patches=total_patches + 1)
            assert len(patches) == 1  # Should return one patch
            assert patches[0].shape == (1, 64, 64)
            break  # Test only one label

if __name__ == '__main__':
    pytest.main()