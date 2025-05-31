import os
import torch
from torch.utils.data import Dataset


class KittiDisparity(Dataset):
    """Main Class for Image Folder loader."""

    def __init__(self,  loaded_in_memory=False,
                 filelist=None, image_loader=None, target_loader=None,
                 training=True,
                 co_transforms=None,
                 input_transforms=None,
                 target_transforms=None,
                 return_filenames=False
                 ):
        """Init function."""

        self.loaded_in_memory = loaded_in_memory
        self.imgs = filelist
        self.training = training

        # data augmentation
        self.co_transforms = co_transforms
        self.input_transforms = input_transforms
        self.target_transforms = target_transforms

        # loaders
        self.image_loader = image_loader
        self.target_loader = target_loader

        # return filenames or not
        self.return_filenames = return_filenames

    def __getitem__(self, index):
        """Get item."""

        input_path = self.imgs[index][0]
        target_path = self.imgs[index][1]
        img = apply_function_list(input_path, self.image_loader)
        target = apply_function_list(target_path, self.target_loader)

        # apply co transforms
        if self.co_transforms is not None:
            img, target = self.co_transforms(img, target)

        # apply transforms for inputs
        if self.input_transforms is not None:
            img = apply_function_list(img, self.input_transforms)

        # apply transform for targets
        if self.target_transforms is not None:
            target = apply_function_list(target, self.target_transforms)

        if self.return_filenames:
            return img, target, self.imgs[index][0]
        else:
            return img, target

    def __len__(self):
        """Length."""
        return len(self.imgs)


def apply_function_list(x, fun):
    """Apply a function or list over a list of object, or single object."""
    if isinstance(x, list):
        y = []
        if isinstance(fun, list):
            for x_id, x_elem in enumerate(x):
                if (fun[x_id] is not None) and (x_elem is not None):
                    y.append(fun[x_id](x_elem))
                else:
                    y.append(None)
        else:
            for x_id, x_elem in enumerate(x):
                if x_elem is not None:
                    y.append(fun(x_elem))
                else:
                    y.append(None)
    else:
        y = fun(x)

    return y


def load_filenames(path_data, path_target, data_dir=None, target_dir=None):
    if target_dir is None:
        target_dir = data_dir

    filenames_data = open(path_data, "r").readlines()
    filenames_target = open(path_target, "r").readlines()
    assert len(filenames_data) == len(filenames_target)

    if data_dir is not None:
        filenames = [[
            [os.path.join(data_dir, g.split("\n")[0])
             for g in f[0].split(" ")],
            [os.path.join(target_dir, g.split("\n")[0])
             for g in f[1].split(" ")],
        ] for f in zip(filenames_data, filenames_target)]
    else:
        filenames = [[
            [g.split("\n")[0] for g in f[0].split(" ")],
            [g.split("\n")[0] for g in f[1].split(" ")],
        ] for f in zip(filenames_data, filenames_target)]
    return filenames
