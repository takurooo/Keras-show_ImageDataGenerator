"""
IMG_FILEに指定されている画像ファイルに変換をかけた後にrow * col表示します。
IMG_FILE変数に画像のファイルパスを設定してください。
"""
#-------------------------------------
# imports
#-------------------------------------
import os
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

#-------------------------------------
# defines
#-------------------------------------
'''
画像のファイルパスを指定してください。
'''
IMG_FILE = r"/Users/xxxx/xxx.jpg"

#-------------------------------------
# functions
#-------------------------------------
def show_imgs(imgs, row, col):
    """Show PILimages as row*col
     # Arguments
            imgs: 1-D array, include PILimages
            row: Int, row for plt.subplot
            col: Int, column for plt.subplot
    """
    if len(imgs) != (row * col):
        raise ValueError("Invalid imgs len:{} col:{} row:{}".format(len(imgs), row, col))

    for i, img in enumerate(imgs):
        plot_num = i+1
        plt.subplot(row, col, plot_num)
        plt.tick_params(labelbottom="off") # x軸の削除
        plt.tick_params(labelleft="off") # y軸の削除
        plt.imshow(img)
    plt.show()


def main():

    img_path = IMG_FILE
    # 指定されたファイルがなかったら例外発生
    if not os.path.exists(img_path):
        raise ValueError("Invalid img_path: ", img_path)
    # 画像ファイルをPIL形式でオープン
    img = image.load_img(img_path)
    # PIL形式の画像をndarray形式に変換 for datagen.flow
    x = image.img_to_array(img)
    # (height, width, 3) -> (1, height, width, 3) for datagen.flow
    x = x.reshape((1,) + x.shape)

    datagen = ImageDataGenerator(featurewise_center=False,
                                 samplewise_center=False,
                                 fill_mode='nearest',
                                 rotation_range=0,
                                 width_shift_range=0,
                                 height_shift_range=0,
                                 shear_range=0,
                                 zoom_range=0,
                                 horizontal_flip=False,
                                 vertical_flip=False,
                                 rescale=0
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

#-------------------------------------
# main functions
#-------------------------------------
if __name__ == '__main__':
    main()
