# Keras-show_ImageDataGenerator
This script shows image files that converted using `keras.preprocessing.image.ImageDataGenerator`

# Usage
1 Creat the environment using `pipenv sync`.  
2 Set the configuration parameters in `config.json`.  
3 Run the command `python show_generator config.json`.  
4 The script will show 16 images.

Note: If you want to change the number of images shown, you can modify the ROW and COL parameters in the show_generator.py file. By default, the script shows 16 images.

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
