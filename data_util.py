
from os.path import join
import random

from PIL import Image, ImageOps

from torch.utils.data import Dataset



def modcrop(img, scale_factor):
#保证图片大小是2的倍数，以便后面对图片下采样2倍得到lr
    (ih, iw) = img.size
    ih = ih - (ih % scale_factor)
    iw = iw - (iw % scale_factor)
    img = img.crop((0, 0, ih, iw))

    return img


def load_img(hr_file_path, scale_factor):
    # 加载图片
    target_img = modcrop(Image.open(hr_file_path).convert('RGB'),4)#读取作为目标的高分辨率图像并保证大小能被2整除
    downsampled_tar_img = target_img.resize(
        (int(target_img.size[0] / scale_factor), int(target_img.size[1] / scale_factor)), Image.BICUBIC)#下采样两倍获得作为输入的低分辨率图像


    return target_img, downsampled_tar_img


def get_patch(img_in, img_tar,  patch_size_w, patch_size_h,scale_factor, ix=-1, iy=-1):
    # 截取训练图片块，大小为（patch_size_w, patch_size_h）
    '''
    :param img_in:              downsamped version of target image
    :param img_tar:             target image
    :param imgs:                downsampled images
    :param patch_size:          size of image patch
    :param scale_factor:
    :param nFrames:
    :param ix:                  start point of x
    :param iy:                  start point of y
    :return:                    image patches
    '''

    (ih, iw) = img_in.size
    (th, tw) = (ih * scale_factor, iw * scale_factor)

    patch_mult = scale_factor
    tp_w = patch_size_w * patch_mult
    tp_h = patch_size_h * patch_mult
    ip_w = tp_w //scale_factor
    ip_h = tp_h // scale_factor

    if ix == -1:          # 随机选择起始点
        ix = random.randrange(0, iw - ip_h + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip_w + 1)

    (tx, ty) = (scale_factor * ix, scale_factor * iy)

    img_in = img_in.crop((iy, ix, iy + ip_w, ix + ip_h))
    img_tar = img_tar.crop((ty, tx, ty + tp_w, tx + tp_h))

    return img_in, img_tar


class DatasetFromFolder(Dataset):

    def __init__(self, hr_image_dir, upscale_factor,
                 file_list, get_patch,patch_size_w, patch_size_h,transform=None):
        super(DatasetFromFolder, self).__init__()

        hr_alist = [line.rstrip() for line in open(join(hr_image_dir,file_list).replace('\\','/'))]
        self.hr_filenames=[join(hr_image_dir, x).replace('\\','/') for x in hr_alist]

        self.upscale_factor = upscale_factor
        self.transform = transform
        self.patch_size_w = patch_size_w
        self.patch_size_h = patch_size_h
        self.get=get_patch

    def __getitem__(self, index):
        #加载图片
        target_img, downsampled_tar_img = load_img(self.hr_filenames[index],self.upscale_factor)
        #从高分辨率图像及低分辨率图像中截取图像块作为训练的图像对
        if self.get:
            downsampled_tar_img, target_img = get_patch(downsampled_tar_img, target_img,
                                                                       self.patch_size_w, self.patch_size_h, self.upscale_factor)

        if self.transform:
            target = self.transform(target_img)
            imgs = self.transform(downsampled_tar_img)

        return target,  imgs


    def __len__(self):
        return len(self.hr_filenames)-1