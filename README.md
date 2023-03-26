# Keras-show_ImageDataGenerator
Show image file converted by keras.preprocessing.image.ImageDataGenerator

# Usage
1 Set config parameters in `config.json`.  
3 `python show_generator`  
4 Script show 16 images.


## config.json
```json
{
    "img_path": "imgs/sample.jpg",
    "featurewise_center": "False",
    "samplewise_center": "False",
    "fill_mode": "nearest",
    "rotation_range": 0,
    "width_shift_range": 0,
    "height_shift_range": 0,
    "shear_range": 0,
    "zoom_range": 0,
    "horizontal_flip": false,
    "vertical_flip": false,
    "rescale": 0
}
```

## raw_image
![raw_image](https://github.com/takurooo/Keras-show_ImageDataGenerator/blob/images/dog.jpg?raw=true)

## height_shift_range(0.5)
![height_shift_range](https://github.com/takurooo/Keras-show_ImageDataGenerator/blob/images/height_0.5.png?raw=true)

## width_shift_range(0.5)
![width_shift_range](https://github.com/takurooo/Keras-show_ImageDataGenerator/blob/images/widht_0.5.png?raw=true)

## horizontal_flip(True)
![horizontal_flip](https://github.com/takurooo/Keras-show_ImageDataGenerator/blob/images/horizontal_flip.png?raw=true)

## vertical_flip(True)
![vertical_flip](https://github.com/takurooo/Keras-show_ImageDataGenerator/blob/images/vertical_flip.png?raw=true)

## rotation_range(90)
![rotation_range](https://github.com/takurooo/Keras-show_ImageDataGenerator/blob/images/rotation_range_90.png?raw=true)

## shear_range(40)
![shear](https://github.com/takurooo/Keras-show_ImageDataGenerator/blob/images/shear_40.png?raw=true)

## zoom_range([0.5,1])
![shear](https://github.com/takurooo/Keras-show_ImageDataGenerator/blob/images/zoom_0.5_1.png?raw=true)

## zoom_range([1,1.5])
![shear](https://github.com/takurooo/Keras-show_ImageDataGenerator/blob/images/zoom_1_1.5.png?raw=true)
