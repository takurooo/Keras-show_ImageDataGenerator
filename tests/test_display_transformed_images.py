import json

import matplotlib.pyplot as plt
import pytest
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from display_transformed_images import (
    create_image_data_generator,
    display_images_in_grid,
    generate_transformed_images,
    load_json,
    main,
)


@pytest.fixture
def tmp_test_image(tmp_path):
    img = Image.new("RGB", (50, 50), color=(73, 94, 107))
    img_path = tmp_path / "test_image.jpg"
    img.save(img_path)
    return img_path


def test_load_json_valid_file(tmp_path):
    # Create a temporary JSON file with valid data
    valid_json_file = tmp_path / "valid.json"
    valid_data = {"key": "value"}
    with open(valid_json_file, "w", encoding="utf-8") as f:
        json.dump(valid_data, f)

    result = load_json(valid_json_file)
    assert result == valid_data


def test_load_json_invalid_file(tmp_path):
    # Create a temporary JSON file with invalid data
    invalid_json_file = tmp_path / "invalid.json"
    with open(invalid_json_file, "w", encoding="utf-8") as f:
        f.write("Invalid JSON data")

    with pytest.raises(ValueError, match=r"Invalid JSON format in file .*"):
        load_json(invalid_json_file)


def test_load_json_nonexistent_file():
    non_existent_file = "non_existent_file.json"
    with pytest.raises(ValueError, match=r"File not found: .*"):
        load_json(non_existent_file)


def test_create_image_data_generator_valid_config():
    valid_config = {
        "featurewise_center": False,
        "samplewise_center": False,
        "fill_mode": "nearest",
        "rotation_range": 0,
        "width_shift_range": 0.0,
        "height_shift_range": 0.0,
        "shear_range": 0.0,
        "zoom_range": 0.0,
        "horizontal_flip": False,
        "vertical_flip": False,
        "rescale": 0,
    }
    result = create_image_data_generator(valid_config)
    assert isinstance(result, ImageDataGenerator)


def test_create_image_data_generator_invalid_config():
    invalid_config = {"rotation_range": 0, "invalid_key": 0}
    with pytest.raises(ValueError, match=r"Invalid config key or value in .*"):
        create_image_data_generator(invalid_config)


def test_generate_transformed_images_valid_args(tmp_test_image):
    datagen = ImageDataGenerator(rotation_range=40)
    num_imgs = 5
    imgs = generate_transformed_images(datagen, tmp_test_image, num_imgs)
    assert len(imgs) == num_imgs
    assert all(isinstance(img, Image.Image) for img in imgs)


def test_generate_transformed_images_invalid_img_path():
    datagen = ImageDataGenerator(rotation_range=40)
    invalid_img_path = "non_existent_image.jpg"
    num_imgs = 5
    with pytest.raises(ValueError, match=r"Invalid image_path: .* doesn't exist"):
        generate_transformed_images(datagen, invalid_img_path, num_imgs)


def test_generate_transformed_images_invalid_num_imgs(tmp_test_image):
    datagen = ImageDataGenerator(rotation_range=40)
    invalid_num_imgs = 0
    with pytest.raises(ValueError, match=r"num_imgs must be a positive integer\."):
        generate_transformed_images(datagen, tmp_test_image, invalid_num_imgs)


def test_display_images_in_grid_valid_args(mocker, tmp_test_image):
    row, col = 2, 3
    valid_img_num = row * col
    imgs = [tmp_test_image] * valid_img_num
    mocker.patch.object(plt, "imshow")
    mocker.patch.object(plt, "show")
    display_images_in_grid(imgs, row, col)
    plt.show.imshow()
    plt.show.assert_called_once()


def test_display_images_in_grid_invalid_args():
    row, col = 2, 3
    invalid_img_num = row * col + 1
    imgs = [tmp_test_image] * invalid_img_num
    with pytest.raises(ValueError, match=r"Invalid imgs len:.* col:.* row:.*"):
        display_images_in_grid(imgs, row, col)


def test_main_valid_args(mocker):
    mocked_json_data = {"rescale": 0}
    load_json_mock = mocker.patch(
        "display_transformed_images.load_json", return_value=mocked_json_data
    )
    create_image_data_generator_mock = mocker.patch(
        "display_transformed_images.create_image_data_generator"
    )
    generate_transformed_images_mock = mocker.patch(
        "display_transformed_images.generate_transformed_images"
    )
    display_images_in_grid_mock = mocker.patch(
        "display_transformed_images.display_images_in_grid"
    )

    config_path = "test_config.json"
    img_path = "test_image.jpg"
    row, col = 4, 4
    main(config_path, img_path, row, col)

    load_json_mock.assert_called_once_with(config_path)
    create_image_data_generator_mock.assert_called_once_with(mocked_json_data)
    generate_transformed_images_mock.assert_called_once_with(
        create_image_data_generator_mock.return_value, img_path, row * col
    )
    display_images_in_grid_mock.assert_called_once_with(
        generate_transformed_images_mock.return_value, row, col
    )
