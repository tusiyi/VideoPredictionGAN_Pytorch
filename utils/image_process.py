import math
import os
from PIL import Image


def central_crop(image, new_h, new_w):
    """
    对输入图像做中心裁剪
    :param image: PIL.Image格式的输入图像
    :param new_h: 裁剪后的h
    :param new_w: 裁剪后的w
    :return: 裁剪后的PIL.Image格式图像
    """
    w, h = image.size
    top = (h - new_h) // 2
    bottom = (h + new_h) // 2
    left = (w - new_w) // 2
    right = (w + new_w) // 2
    crop_img = image.crop((left, top, right, bottom))
    return crop_img


if __name__ == '__main__':

    # central crop KITTI_raw_data(left, RGB: image_02)
    main_dir = '/media/tsy/F/KITTI_raw_data'
    save_dir = '/media/tsy/F/KITTI_left_RGB'
    crop_h, crop_w = 360, 450
    for root, dirs, files in os.walk(main_dir):
        for _dir in dirs:
            # if _dir == '2011_09_26':
            #     continue
            image_dir = os.path.join(main_dir, _dir, 'image_02/data')
            images = os.listdir(image_dir)
            for name in images:
                img_path = os.path.join(image_dir, name)
                img = Image.open(img_path)
                cropped = central_crop(img, crop_h, crop_w)
                if not os.path.exists(os.path.join(save_dir, _dir)):
                    os.mkdir(os.path.join(save_dir, _dir))
                cropped.save(os.path.join(save_dir, _dir, name))
            print(f'{_dir} process finished.')
        break  # 否则root会进入下一级目录
