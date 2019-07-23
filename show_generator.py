
# -------------------------------------
# imports
# -------------------------------------
import os
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import json

# -------------------------------------
# defines
# -------------------------------------

JSON_PATH = os.path.join(os.getcwd(), 'args.json')

# -------------------------------------
# functions
# -------------------------------------


def get_args():
    with open(JSON_PATH, "r") as f:
        j = json.load(f)
    return j


def show_imgs(imgs, row, col):
    """Show PILimages as row*col
     # Arguments
            imgs: 1-D array, include PILimages
            row: Int, row for plt.subplot
            col: Int, column for plt.subplot
    """
    if len(imgs) != (row * col):
        raise ValueError(
            "Invalid imgs len:{} col:{} row:{}".format(len(imgs), row, col))

    for i, img in enumerate(imgs):
        plot_num = i+1
        plt.subplot(row, col, plot_num)
        plt.tick_params(labelbottom=False)  # x軸の削除
        plt.tick_params(labelleft=False)  # y軸の削除
        plt.imshow(img)
    plt.show()


def main(args):

    img_path = args["img_path"]
    featurewise_center = args["featurewise_center"]
    samplewise_center = args["samplewise_center"]
    fill_mode = args["fill_mode"]
    rotation_range = args["rotation_range"]
    width_shift_range = args["width_shift_range"]
    height_shift_range = args["height_shift_range"]
    shear_range = args["shear_range"]
    zoom_range = args["zoom_range"]
    horizontal_flip = args["horizontal_flip"] == "True"
    vertical_flip = args["vertical_flip"] == "True"
    rescale = args["rescale"]

    # 指定されたファイルがなかったら例外発生
    if not os.path.exists(img_path):
        raise ValueError("Invalid img_path: ", img_path)
    # 画像ファイルをPIL形式でオープン
    img = image.load_img(img_path)
    # PIL形式の画像をndarray形式に変換 for datagen.flow
    x = image.img_to_array(img)
    # (height, width, 3) -> (1, height, width, 3) for datagen.flow
    x = x.reshape((1,) + x.shape)

    datagen = ImageDataGenerator(featurewise_center=featurewise_center,
                                 samplewise_center=samplewise_center,
                                 fill_mode=fill_mode,
                                 rotation_range=rotation_range,
                                 width_shift_range=width_shift_range,
                                 height_shift_range=height_shift_range,
                                 shear_range=shear_range,
                                 zoom_range=zoom_range,
                                 horizontal_flip=horizontal_flip,
                                 vertical_flip=vertical_flip,
                                 rescale=rescale
                                 )

    max_img_num = 16
    imgs = []
    for d in datagen.flow(x, batch_size=1):
        # このあと画像を表示するためにndarrayをPIL形式に変換して保存する
        imgs.append(image.array_to_img(d[0], scale=True))
        # datagen.flowは無限ループするため必要な枚数取得できたらループを抜ける
        if (len(imgs) % max_img_num) == 0:
            break

    show_imgs(imgs, row=4, col=4)


# -------------------------------------
# main functions
# -------------------------------------
if __name__ == '__main__':
    main(get_args())
