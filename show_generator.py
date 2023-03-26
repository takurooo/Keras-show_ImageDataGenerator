"""
This script is designed to apply various image augmentations to a given image
and display the transformed images in a grid. The image augmentations are
configured through a JSON configuration file. The script uses Keras'
ImageDataGenerator for applying the transformations and matplotlib for
displaying the images.
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, image_utils


ROW = 4
COL = 4
IMAGE_NUM = ROW * COL


def get_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("config_path", type=str, help="your config file path")
    parser.add_argument("img_path", type=str, help="your image file path")
    return parser.parse_args()


def load_config(config_path):
    """
    Load configuration from a JSON file.

    Args:
        config_path (str): Path to the configuration JSON file.

    Returns:
        dict: A dictionary containing configuration settings.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        j = json.load(f)
    return j


def create_datagen(config):
    """
    Create an ImageDataGenerator instance based on the given configuration.

    Args:
        config (dict): A dictionary containing the configuration settings for ImageDataGenerator.

    Returns:
        ImageDataGenerator: An instance of the ImageDataGenerator class.
    """
    return ImageDataGenerator(
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


def generate_images(datagen, img_path, num_images):
    """
    Generate a specified number of images using the given ImageDataGenerator and image path.

    Args:
        datagen (ImageDataGenerator): An instance of the ImageDataGenerator class.
        img_path (str): Path to the input image file.
        num_images (int): Number of images to generate.

    Returns:
        list: A list of generated PIL images.
    """
    # open the image file as PIL format
    img = image_utils.load_img(img_path)
    # convert PIL format to ndarray format for datagen.flow
    img_array = image_utils.img_to_array(img)
    # (height, width, 3) -> (1, height, width, 3) for datagen.flow
    x = img_array.reshape((1,) + img_array.shape)

    imgs = []
    for d in datagen.flow(x, batch_size=1):
        # convert ndarray format to PIL format to display the image.
        imgs.append(image_utils.array_to_img(d[0], scale=True))
        # since datagen.flow loops infinitely, you need to break out of the loop
        # once you have obtained the required number of images.
        if (len(imgs) % num_images) == 0:
            break
    return imgs


def show_imgs(imgs, row, col):
    """
    Display images in a grid format using matplotlib.

    Args:
        imgs (list): A list of PIL images to display.
        row (int): Number of rows in the grid.
        col (int): Number of columns in the grid.

    Raises:
        ValueError: If the number of images does not match the grid size (row * col).
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


def main(config_path, img_path):
    """
    Main function.

    Args:
        config_path (str): The path to the configuration file.
        img_path (str): The path to the image file.
    """
    if not os.path.exists(img_path):
        raise ValueError(f"Invalid img_path: {img_path}")

    config = load_config(config_path)
    datagen = create_datagen(config)
    imgs = generate_images(datagen, img_path, IMAGE_NUM)
    show_imgs(imgs, ROW, COL)


if __name__ == "__main__":
    args = get_args()
    main(args.config_path, args.img_path)
