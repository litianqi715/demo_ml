"""
    loader for open_images_v6

1.optimized for person cls   
"""
import numpy as np
import collections

class OpenV6Loader(object):
    def __init__(self,
        root,
        split="training",
        is_transform=False,
        img_size=[1024, 768],
        augmentations=None,
        img_norm=True,
        test_mode=False
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.test_mode = test_mode
        self.n_classes = 150
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = collections.defaultdict(list)

        if not self.test_mode:
            for split in ["training", "validation"]:
                file_list = recursive_glob(
                    rootdir=self.root + "images/" + self.split + "/", suffix=".jpg"
                )
                self.files[split] = file_list

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

