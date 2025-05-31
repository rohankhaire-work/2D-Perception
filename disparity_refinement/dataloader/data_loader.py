import numpy as np
import cv2
from PIL import Image


def input_rgb_image_loader(data_path):
    """Load an image.
        Args:
            image_path
        Returns:
            A numpy float32 array shape (w,h, n_channel)
    """

    # Load .png RGB image
    im = np.array(Image.open(data_path)).astype(np.float32)

    # Resize
    size = (512, 256)
    im = cv2.resize(
        im, size, interpolation=cv2.INTER_LINEAR).astype(np.float32)

    # Normalize between 0 - +1
    im /= 255.

    return im


def input_sgm(data_path):

    # Load disp map
    im = cv2.imread(data_path, cv2.IMREAD_ANYDEPTH)
    h, w = im.shape
    max_disp = 192

    # Convert to float
    disp_float = im.astype(np.float32) / 256.0

    # Resize
    size = (512, 256)
    disp_resized = cv2.resize(disp_float, size,
                              interpolation=cv2.INTER_NEAREST)

    # Normalize disparity
    disp_norm = np.clip(disp_resized, 0, max_disp) / max_disp
    disp_norm = disp_norm.reshape(disp_norm.shape+(1,))

    return disp_norm


def train_target_gt_depth_loader(data_path):

    # Load disp map
    im = cv2.imread(data_path, cv2.IMREAD_ANYDEPTH)

    # Convert to float
    disp_float = im.astype(np.float32) / 256.0

    # Resize
    size = (512, 256)
    disp_resized = cv2.resize(
        disp_float, size, interpolation=cv2.INTER_NEAREST)

    disp_resized = disp_resized.reshape(disp_resized.shape+(1,))

    return disp_resized


def test_target_gt_depth_loader(data_path):

    # Load disp map
    im = cv2.imread(data_path, cv2.IMREAD_ANYDEPTH)

    # Convert to float
    disp_float = im.astype(np.float32) / 256.0

    # Resize
    size = (512, 256)
    disp_resized = cv2.resize(
        disp_float, size, interpolation=cv2.INTER_NEAREST)

    disp_resized = disp_resized.reshape(disp_resized.shape+(1,))

    return disp_resized
