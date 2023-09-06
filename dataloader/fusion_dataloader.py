import pathlib

import cv2
import kornia.utils
import torch.utils.data
import torchvision.transforms.functional
from PIL import Image
import numpy as np


class FuseTrainData(torch.utils.data.Dataset):
    """
    Load dataset with infrared folder path and visible folder path
    """

    # TODO: remove ground truth reference
    def __init__(self, ir_folder: pathlib.Path, vi_folder: pathlib.Path, gt_folder: pathlib.Path, ir_map: pathlib.Path, vi_map: pathlib.Path, crop=lambda x: x):
        super(FuseTrainData, self).__init__()

        self.crop = crop
        # gain infrared and visible images list
        self.ir_list = [x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.vi_list = [x for x in sorted(vi_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.gt_list = [x for x in sorted(gt_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]

        self.ir_map_list = [x for x in sorted(ir_map.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.vi_map_list = [x for x in sorted(vi_map.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]

    def __getitem__(self, index):
        # gain image path
        ir_path = self.ir_list[index]
        vi_path = self.vi_list[index]
        gt_path = self.gt_list[index]

        ir_map_path = self.ir_map_list[index]
        vi_map_path = self.vi_map_list[index]

        assert ir_path.name == vi_path.name, f"Mismatch ir:{ir_path.name} vi:{vi_path.name}."

        # read image as type Tensor
        ir = self.imread(path=ir_path, flags=cv2.IMREAD_GRAYSCALE)
        vi = self.imread_rgb2ycrcb(path=vi_path, flags=cv2.IMREAD_COLOR)
        gt = self.imread(path=gt_path, flags=cv2.IMREAD_GRAYSCALE)

        # crop same patch
        # patch = torch.cat([ir, vi], dim=0)
        # patch = torchvision.transforms.functional.to_pil_image(patch)
        # patch = self.crop(patch)
        # patch = torchvision.transforms.functional.to_tensor(patch)
        # ir, vi = torch.chunk(patch, 2, dim=0)


        ir_map = self.imread(path=ir_map_path, flags=cv2.IMREAD_GRAYSCALE)
        vi_map = self.imread(path=vi_map_path, flags=cv2.IMREAD_GRAYSCALE)

        return (ir, vi, gt), (str(ir_path), str(vi_path)), (ir_map, vi_map)

    def __len__(self):
        return len(self.ir_list)

    @staticmethod
    def imread(path: pathlib.Path, flags=cv2.IMREAD_GRAYSCALE):
        im_cv = cv2.imread(str(path), flags)
        assert im_cv is not None, f"Image {str(path)} is invalid."
        im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
        return im_ts

    @staticmethod
    def imread_vi(path: pathlib.Path):
        im_cv = np.array(Image.open(path))
        image_cv = np.asarray(Image.fromarray(im_cv), dtype=np.float32).transpose((2, 0, 1))/ 255.0
        return torch.tensor(image_cv)

    @staticmethod
    def imread_rgb2ycrcb(path: pathlib.Path, flags=cv2.IMREAD_COLOR):
        im_cv = cv2.imread(str(path), flags)
        vi_ycbcr = cv2.cvtColor(im_cv, cv2.COLOR_BGR2YCrCb)
        vi_y = vi_ycbcr[:, :, 0]
        vi_cb = vi_ycbcr[:, :, 1]
        vi_cr = vi_ycbcr[:, :, 2]
        vi_y  = kornia.utils.image_to_tensor(vi_y / 255.).type(torch.FloatTensor)
        vi_cb = kornia.utils.image_to_tensor(vi_cb / 255.).type(torch.FloatTensor)
        vi_cr = kornia.utils.image_to_tensor(vi_cr / 255.).type(torch.FloatTensor)
        vi = torch.cat([vi_y, vi_cb, vi_cr], dim=0)
        return vi


class FuseEdgeTrainData(torch.utils.data.Dataset):
    """
    Load dataset with infrared folder path and visible folder path
    """

    # TODO: remove ground truth reference
    def __init__(self, ir_folder: pathlib.Path, vi_folder: pathlib.Path, gt_folder: pathlib.Path, edge_folder: pathlib.Path, ir_map: pathlib.Path, vi_map: pathlib.Path, crop=lambda x: x):
        super(FuseEdgeTrainData, self).__init__()

        self.crop = crop
        # gain infrared and visible images list
        self.ir_list = [x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.vi_list = [x for x in sorted(vi_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.gt_list = [x for x in sorted(gt_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.edge_list = [x for x in sorted(edge_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]

        self.ir_map_list = [x for x in sorted(ir_map.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.vi_map_list = [x for x in sorted(vi_map.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]

    def __getitem__(self, index):
        # gain image path
        ir_path = self.ir_list[index]
        vi_path = self.vi_list[index]
        gt_path = self.gt_list[index]
        edge_path = self.edge_list[index]

        ir_map_path = self.ir_map_list[index]
        vi_map_path = self.vi_map_list[index]

        assert ir_path.name == vi_path.name, f"Mismatch ir:{ir_path.name} vi:{vi_path.name}."

        # read image as type Tensor
        ir = self.imread(path=ir_path, flags=cv2.IMREAD_GRAYSCALE)
        vi = self.imread_rgb2ycrcb(path=vi_path, flags=cv2.IMREAD_COLOR)
        gt = self.imread(path=gt_path, flags=cv2.IMREAD_GRAYSCALE)
        edge = self.imread(path=edge_path, flags=cv2.IMREAD_GRAYSCALE)

        # crop same patch
        # patch = torch.cat([ir, vi], dim=0)
        # patch = torchvision.transforms.functional.to_pil_image(patch)
        # patch = self.crop(patch)
        # patch = torchvision.transforms.functional.to_tensor(patch)
        # ir, vi = torch.chunk(patch, 2, dim=0)


        ir_map = self.imread(path=ir_map_path, flags=cv2.IMREAD_GRAYSCALE)
        vi_map = self.imread(path=vi_map_path, flags=cv2.IMREAD_GRAYSCALE)

        return (ir, vi, gt, edge), (str(ir_path), str(vi_path)), (ir_map, vi_map)

    def __len__(self):
        return len(self.ir_list)

    @staticmethod
    def imread(path: pathlib.Path, flags=cv2.IMREAD_GRAYSCALE):
        im_cv = cv2.imread(str(path), flags)
        assert im_cv is not None, f"Image {str(path)} is invalid."
        im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
        return im_ts

    @staticmethod
    def imread_vi(path: pathlib.Path):
        im_cv = np.array(Image.open(path))
        image_cv = np.asarray(Image.fromarray(im_cv), dtype=np.float32).transpose((2, 0, 1))/ 255.0
        return torch.tensor(image_cv)

    @staticmethod
    def imread_rgb2ycrcb(path: pathlib.Path, flags=cv2.IMREAD_COLOR):
        im_cv = cv2.imread(str(path), flags)
        vi_ycbcr = cv2.cvtColor(im_cv, cv2.COLOR_BGR2YCrCb)
        vi_y = vi_ycbcr[:, :, 0]
        vi_cb = vi_ycbcr[:, :, 1]
        vi_cr = vi_ycbcr[:, :, 2]
        vi_y  = kornia.utils.image_to_tensor(vi_y / 255.).type(torch.FloatTensor)
        vi_cb = kornia.utils.image_to_tensor(vi_cb / 255.).type(torch.FloatTensor)
        vi_cr = kornia.utils.image_to_tensor(vi_cr / 255.).type(torch.FloatTensor)
        vi = torch.cat([vi_y, vi_cb, vi_cr], dim=0)
        return vi


class FuseTestData(torch.utils.data.Dataset):
    """
    Load dataset with infrared folder path and visible folder path
    """

    # TODO: remove ground truth reference
    def __init__(self, ir_folder: pathlib.Path, vi_folder: pathlib.Path, crop=lambda x: x):
        super(FuseTestData, self).__init__()
        self.crop = crop
        # gain infrared and visible images list
        self.ir_list = [x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.vi_list = [x for x in sorted(vi_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]

    def __getitem__(self, index):
        # gain image path
        ir_path = self.ir_list[index]
        vi_path = self.vi_list[index]

        assert ir_path.name == vi_path.name, f"Mismatch ir:{ir_path.name} vi:{vi_path.name}."

        # read image as type Tensor
        ir = self.imread(path=ir_path, flags=cv2.IMREAD_GRAYSCALE)
        # vi = self.imread_vi(path=vi_path)
        vi = self.imread_rgb2ycrcb(path=vi_path, flags=cv2.IMREAD_COLOR)

        # crop same patch
        # patch = torch.cat([ir, vi], dim=0)
        # patch = torchvision.transforms.functional.to_pil_image(patch)
        # patch = self.crop(patch)
        # patch = torchvision.transforms.functional.to_tensor(patch)
        # ir, vi = torch.chunk(patch, 2, dim=0)

        return (ir, vi), (str(ir_path), str(vi_path))

    def __len__(self):
        return len(self.ir_list)

    @staticmethod
    def imread(path: pathlib.Path, flags=cv2.IMREAD_GRAYSCALE):
        im_cv = cv2.imread(str(path), flags)
        assert im_cv is not None, f"Image {str(path)} is invalid."
        im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
        return im_ts

    @staticmethod
    def imread_vi(path: pathlib.Path):
        im_cv = np.array(Image.open(path))
        image_cv = np.asarray(Image.fromarray(im_cv), dtype=np.float32).transpose((2, 0, 1)) / 255.0
        return torch.tensor(image_cv)

    @staticmethod
    def imread_rgb2ycrcb(path: pathlib.Path, flags=cv2.IMREAD_COLOR):
        im_cv = cv2.imread(str(path), flags)
        vi_ycbcr = cv2.cvtColor(im_cv, cv2.COLOR_BGR2YCrCb)
        vi_y = vi_ycbcr[:, :, 0]
        vi_cb = vi_ycbcr[:, :, 1]
        vi_cr = vi_ycbcr[:, :, 2]
        vi_y = kornia.utils.image_to_tensor(vi_y / 255.).type(torch.FloatTensor)
        vi_cb = kornia.utils.image_to_tensor(vi_cb / 255.).type(torch.FloatTensor)
        vi_cr = kornia.utils.image_to_tensor(vi_cr / 255.).type(torch.FloatTensor)
        vi = torch.cat([vi_y, vi_cb, vi_cr], dim=0)
        return vi


