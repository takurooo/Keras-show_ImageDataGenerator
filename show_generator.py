import json
import os

import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, image_utils


CONFIG_PATH = os.path.join(os.getcwd(), "config.json")


def load_config():
    with open(CONFIG_PATH, "r") as f:
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
        raise ValueError(
            "Invalid imgs len:{} col:{} row:{}".format(len(imgs), row, col)
        )

    for i, img in enumerate(imgs):
        plot_num = i + 1
        plt.subplot(row, col, plot_num)
        plt.tick_params(labelbottom=False)  # remove x axis
        plt.tick_params(labelleft=False)  # remove y axis
        plt.imshow(img)
    plt.show()


def main(config):
    img_path = config["img_path"]
    if not os.path.exists(img_path):
        raise ValueError("Invalid img_path: ", img_path)

    # open the image file as PIL format
    img = image_utils.load_img(img_path)
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

    max_img_num = 16
    imgs = []
    for d in datagen.flow(x, batch_size=1):
        # convert ndarray format to PIL format to display the image.
        imgs.append(image_utils.array_to_img(d[0], scale=True))
        # since datagen.flow loops infinitely, you need to break out of the loop
        # once you have obtained the required number of images.
        if (len(imgs) % max_img_num) == 0:
            break

    show_imgs(imgs, row=4, col=4)


if __name__ == "__main__":
    main(load_config())
