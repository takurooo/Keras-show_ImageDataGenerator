import argparse
import json
import os

import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, image_utils


ROW = 4
COL = 4
IMAGE_NUM = ROW * COL


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("config_path", type=str, help="your config file path")
    parser.add_argument("img_path", type=str, help="your image file path")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        j = json.load(f)
    return j


def show_imgs(imgs, row, col):
    """Show PIL format images as row*col
    # Arguments
           imgs: 1-D array, include PILimages
           row: Int, row for plt.subplot
           col: Int, column for plt.subplot
    """
    if len(imgs) != (row * col):
        raise ValueError(f"Invalid imgs len:{len(imgs)} col:{row} row:{col}")

    for i, img in enumerate(imgs):
        plot_num = i + 1
        plt.subplot(row, col, plot_num)
        plt.tick_params(labelbottom=False)  # remove x axis
        plt.tick_params(labelleft=False)  # remove y axis
        plt.imshow(img)
    plt.show()


def main(args):
    config = load_config(args.config_path)
    if not os.path.exists(args.img_path):
        raise ValueError(f"Invalid img_path: {args.img_path}")

    # open the image file as PIL format
    img = image_utils.load_img(args.img_path)
    # convert PIL format to ndarray format for datagen.flow
    x = image_utils.img_to_array(img)
    # (height, width, 3) -> (1, height, width, 3) for datagen.flow
    x = x.reshape((1,) + x.shape)

    datagen = ImageDataGenerator(
        featurewise_center=config["featurewise_center"],
        samplewise_center=config["samplewise_center"],
        fill_mode=config["fill_mode"],
        rotation_range=config["rotation_range"],
        width_shift_range=config["width_shift_range"],
        height_shift_range=config["height_shift_range"],
        shear_range=config["shear_range"],
        zoom_range=config["zoom_range"],
        horizontal_flip=config["horizontal_flip"],
        vertical_flip=config["vertical_flip"],
        rescale=config["rescale"],
    )

    imgs = []
    for d in datagen.flow(x, batch_size=1):
        # convert ndarray format to PIL format to display the image.
        imgs.append(image_utils.array_to_img(d[0], scale=True))
        # since datagen.flow loops infinitely, you need to break out of the loop
        # once you have obtained the required number of images.
        if (len(imgs) % IMAGE_NUM) == 0:
            break

    show_imgs(imgs, ROW, COL)


if __name__ == "__main__":
    main(get_args())
