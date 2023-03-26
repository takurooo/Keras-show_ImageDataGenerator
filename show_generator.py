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


def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the JSON configuration file containing the image transformation settings.",
    )
    parser.add_argument(
        "img_path",
        type=str,
        help="Path to the input image file to apply transformations.",
    )
    return parser.parse_args()


def load_image_data_generator_config(config_path):
    """
    Load configuration from a JSON file.

    Args:
        config_path (str): Path to the configuration JSON file.

    Raises:
        ValueError: If there is an error decoding the JSON format or the file is not found.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_json = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in file {config_path}") from e
    except FileNotFoundError as e:
        raise ValueError(f"File not found: {config_path}") from e
    return config_json


def create_image_data_generator(config):
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


def generate_transformed_images(datagen, img_path, num_imgs):
    """
    Generate a specified number of images using the given ImageDataGenerator and image path.

    Args:
        datagen (ImageDataGenerator): An instance of the ImageDataGenerator class.
        img_path (str): Path to the input image file.
        num_imgs (int): Number of images to generate.

    Returns:
        list: A list of generated PIL images.
    """
    # open the image file as PIL format
    img = image_utils.load_img(img_path)
    # convert PIL format to ndarray format for datagen.flow
    img_array = image_utils.img_to_array(img)
    # (height, width, 3) -> (1, height, width, 3) for datagen.flow
    img_array_reshaped = img_array.reshape((1,) + img_array.shape)

    imgs = []
    for transformed_img in datagen.flow(img_array_reshaped, batch_size=1):
        # convert ndarray format to PIL format to display the image.
        imgs.append(image_utils.array_to_img(transformed_img[0], scale=True))
        # since datagen.flow loops infinitely, you need to break out of the loop
        # once you have obtained the required number of images.
        if (len(imgs) % num_imgs) == 0:
            break
    return imgs


def display_images_in_grid(imgs, row, col):
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


def main(config_path, img_path, row=4, col=4):
    """
    Main function.

    Args:
        config_path (str): The path to the configuration file.
        image_path (str): The path to the image file.
        row (int): Number of rows in the grid. Defaults to 4.
        col (int): Number of columns in the grid. Defaults to 4.
    """
    if not os.path.exists(img_path):
        raise ValueError(f"Invalid image_path: {img_path} doesn't exist")
    if not os.path.exists(config_path):
        raise ValueError(f"Invalid config_path: {config_path} doesn't exist")

    config = load_image_data_generator_config(config_path)
    datagen = create_image_data_generator(config)
    imgs = generate_transformed_images(datagen, img_path, row * col)
    display_images_in_grid(imgs, row, col)


if __name__ == "__main__":
    args = parse_args()
    main(args.config_path, args.img_path)
