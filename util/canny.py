# -*- coding: utf-8 -*-
"""
@software: PyCharm
@file: gen_edge.py
@time: 2021/5/10 23:36
"""

import cv2
import os

def Edge_Extract(root):
    img_root = os.path.join(root,'GT')
    edge_root = os.path.join(root,'Edge')

    if not os.path.exists(edge_root):
        os.mkdir(edge_root)

    file_names = os.listdir(img_root)
    img_name = []

    for name in file_names:
        print(f'Generate Edge Image {name} successful!')
        if not name.endswith('.png'):
            assert "This file %s is not PNG"%(name)
        img_name.append(os.path.join(img_root,name[:-4]+'.png'))

    index = 0
    for image in img_name:
        img = cv2.imread(image,0)
        cv2.imwrite(edge_root+'/'+file_names[index],cv2.Canny(img,30,100))
        index += 1
    return 0


def sobel(img, threshold):
    '''
    edge detection based on sobel

    Parameters
    ----------
    img : TYPE
        the image input.
    threshold : TYPE
         varies for application [0 255].

    Returns
    -------
    mag : TYPE
        output after edge detection.

    '''
    G_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    G_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    rows = np.size(img, 0)
    columns = np.size(img, 1)
    mag = np.zeros(img.shape)
    for i in range(0, rows - 2):
        for j in range(0, columns - 2):
            v = sum(sum(G_x * img[i:i + 3, j:j + 3]))  # vertical
            h = sum(sum(G_y * img[i:i + 3, j:j + 3]))  # horizon
            mag[i + 1, j + 1] = np.sqrt((v ** 2) + (h ** 2))

    for p in range(0, rows):
        for q in range(0, columns):
            if mag[p, q] < threshold:
                mag[p, q] = 0
    return mag


if __name__ == '__main__':
    root = './RGBD_for_test/STERE'
    Edge_Extract(root)



def rgb2ycbcr(img: T.Tensor):
    """
    Converts a batch of RGB images to YCbCr.

    :param img:
        a batch of RGB image tensors.
    :return:
        a batch of YCbCr images.
    """

    if len(img.shape) != 4:
        raise ValueError('Input images must have four dimensions, not %d' % len(img.shape))

    Y = 0. + .299 * img[:, 0] + .587 * img[:, 1] + .114 * img[:, 2]
    Cb = 128. - .169 * img[:, 0] - .331 * img[:, 1] + .5 * img[:, 2]
    Cr = 128. + .5 * img[:, 0] - .419 * img[:, 1] - .081 * img[:, 2]
    return T.cat((Y.unsqueeze(1), Cb.unsqueeze(1), Cr.unsqueeze(1)), 1)

def ycbcr2rgb(img: T.Tensor):
    """
    Converts a batch of YCbCr images to RGB.

    :param img:
        a batch of YCbCr image tensors.
    :return:
        a batch of RGB images.
    """

    if len(img.shape) != 4:
        raise ValueError('Input images must have four dimensions, not %d' % len(img.shape))

    R = img[:, 0] + 1.4 * (img[:, 2] - 128.)
    G = img[:, 0] - .343 * (img[:, 1] - 128.) - .711 * (img[:, 2] - 128.)
    B = img[:, 0] + 1.765 * (img[:, 1] - 128.)
    return T.cat((R.unsqueeze(1), G.unsqueeze(1), B.unsqueeze(1)), 1)


def make_lambda_warmup_scheduler(optimizer, warmup_iter, warmup_factor,
                                 max_iter):
    """Reference from:

    https://github.com/dvlab-research/Simple-SR/blob/master/utils/solver.py.
    """
    def lr_lambda(iteration):
        if iteration < warmup_iter:
            return warmup_factor + (1 - warmup_factor) * iteration / warmup_iter
        else:
            return 1 / 2 * (1 + math.cos((iteration - warmup_iter) / (max_iter - warmup_iter) * math.pi))

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    return scheduler


class CannyFilter(nn.Module):
    def __init__(self,
                 k_gaussian=3,
                 mu=0,
                 sigma=1,
                 k_sobel=3,
                 use_cuda=False):
        super(CannyFilter, self).__init__()
        # device
        self.device = 'cuda' if use_cuda else 'cpu'

        # gaussian

        gaussian_2D = get_gaussian_kernel(k_gaussian, mu, sigma)
        self.gaussian_filter = nn.Conv2d(in_channels=1,
                                         out_channels=1,
                                         kernel_size=k_gaussian,
                                         padding=k_gaussian // 2,
                                         bias=False)
        self.gaussian_filter.weight[:] = torch.from_numpy(gaussian_2D)

        # sobel

        sobel_2D = get_sobel_kernel(k_sobel)
        self.sobel_filter_x = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)
        self.sobel_filter_x.weight[:] = torch.from_numpy(sobel_2D)


        self.sobel_filter_y = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)
        self.sobel_filter_y.weight[:] = torch.from_numpy(sobel_2D.T)


        # thin

        thin_kernels = get_thin_kernels()
        directional_kernels = np.stack(thin_kernels)

        self.directional_filter = nn.Conv2d(in_channels=1,
                                            out_channels=8,
                                            kernel_size=thin_kernels[0].shape,
                                            padding=thin_kernels[0].shape[-1] // 2,
                                            bias=False)
        self.directional_filter.weight[:, 0] = torch.from_numpy(directional_kernels)

        # hysteresis

        hysteresis = np.ones((3, 3)) + 0.25
        self.hysteresis = nn.Conv2d(in_channels=1,
                                    out_channels=1,
                                    kernel_size=3,
                                    padding=1,
                                    bias=False)
        self.hysteresis.weight[:] = torch.from_numpy(hysteresis)


    def forward(self, img, low_threshold=None, high_threshold=None, hysteresis=False):
        # set the setps tensors
        B, C, H, W = img.shape
        blurred = torch.zeros((B, C, H, W)).to(self.device)
        grad_x = torch.zeros((B, 1, H, W)).to(self.device)
        grad_y = torch.zeros((B, 1, H, W)).to(self.device)
        grad_magnitude = torch.zeros((B, 1, H, W)).to(self.device)
        grad_orientation = torch.zeros((B, 1, H, W)).to(self.device)

        # gaussian

        for c in range(C):
            blurred[:, c:c+1] = self.gaussian_filter(img[:, c:c+1])

            grad_x = grad_x + self.sobel_filter_x(blurred[:, c:c+1])
            grad_y = grad_y + self.sobel_filter_y(blurred[:, c:c+1])

        # thick edges

        grad_x, grad_y = grad_x / C, grad_y / C
        grad_magnitude = (grad_x ** 2 + grad_y ** 2) ** 0.5
        grad_orientation = torch.atan(grad_y / grad_x)
        grad_orientation = grad_orientation * (360 / np.pi) + 180 # convert to degree
        grad_orientation = torch.round(grad_orientation / 45) * 45  # keep a split by 45

        # thin edges

        directional = self.directional_filter(grad_magnitude)
        # get indices of positive and negative directions
        positive_idx = (grad_orientation / 45) % 8
        negative_idx = ((grad_orientation / 45) + 4) % 8
        thin_edges = grad_magnitude.clone()
        # non maximum suppression direction by direction
        for pos_i in range(4):
            neg_i = pos_i + 4
            # get the oriented grad for the angle
            is_oriented_i = (positive_idx == pos_i) * 1
            is_oriented_i = is_oriented_i + (positive_idx == neg_i) * 1
            pos_directional = directional[:, pos_i]
            neg_directional = directional[:, neg_i]
            selected_direction = torch.stack([pos_directional, neg_directional])

            # get the local maximum pixels for the angle
            is_max = selected_direction.min(dim=0)[0] > 0.0
            is_max = torch.unsqueeze(is_max, dim=1)

            # apply non maximum suppression
            to_remove = (is_max == 0) * 1 * (is_oriented_i) > 0
            thin_edges[to_remove] = 0.0

        # thresholds

        if low_threshold is not None:
            low = thin_edges > low_threshold

            if high_threshold is not None:
                high = thin_edges > high_threshold
                # get black/gray/white only
                thin_edges = low * 0.5 + high * 0.5

                if hysteresis:
                    # get weaks and check if they are high or not
                    weak = (thin_edges == 0.5) * 1
                    weak_is_high = (self.hysteresis(thin_edges) > 1) * weak
                    thin_edges = high * 1 + weak_is_high * 1
            else:
                thin_edges = low * 1


        return blurred, grad_x, grad_y, grad_magnitude, grad_orientation, thin_edges