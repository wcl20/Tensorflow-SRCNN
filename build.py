import cv2
import numpy as np
import os
import random
import shutil
import tqdm
from config import config
from core.io import HDF5Writer
from imutils import paths
from PIL import Image


def main():

    # Build temporary directory to store images
    os.makedirs(config.LR_IMAGES, exist_ok=True)
    os.makedirs(config.HR_IMAGES, exist_ok=True)

    img_paths = list(paths.list_images(config.INPUT_DIR))
    random.shuffle(img_paths)

    print("[INFO] Saving temporary images ...")
    counter = 0
    for img_path in tqdm.tqdm(img_paths):

        image = cv2.imread(img_path)
        # Crop image to fit scale
        height, width = image.shape[:2]
        width -= int(width % config.SCALE)
        height -= int(height % config.SCALE)
        image = image[:height, :width]

        # Down scale image
        low_width = int(width * (1. / config.SCALE))
        low_height = int(height * (1. / config.SCALE))
        lr_image = Image.fromarray(image).resize((low_width, low_height), resample=Image.BICUBIC)
        lr_image = np.array(lr_image)

        # Upscale image using Bicubic interpolation
        high_width = int(low_width * (config.SCALE / 1.))
        high_height = int(low_height * (config.SCALE / 1.))
        hr_image = Image.fromarray(lr_image).resize((high_width, high_height), resample=Image.BICUBIC)
        hr_image = np.array(hr_image)

        # Sliding window
        for y in range(0, height - config.INPUT_DIM + 1, config.STRIDE):
            for x in range(0, width - config.INPUT_DIM + 1, config.STRIDE):

                # Crop input from bicubic image
                input = hr_image[y : y+config.INPUT_DIM, x : x+config.INPUT_DIM]
                # Crop target from high resolution image (original)
                target = image[y+config.PAD : y+config.PAD+config.OUTPUT_DIM, x+config.PAD : x+config.PAD+config.OUTPUT_DIM]
                # Save lr and hr images
                cv2.imwrite(os.path.sep.join([config.LR_IMAGES, f"{counter}.png"]), input)
                cv2.imwrite(os.path.sep.join([config.HR_IMAGES, f"{counter}.png"]), target)

                counter += 1

    print("[INFO] Building HDF5 datasets")
    input_paths = sorted(list(paths.list_images(config.LR_IMAGES)))
    target_paths = sorted(list(paths.list_images(config.HR_IMAGES)))

    input_writer = HDF5Writer(config.LR_HDF5, dims=(len(input_paths), config.INPUT_DIM, config.INPUT_DIM, 3))
    target_writer = HDF5Writer(config.HR_HDF5, dims=(len(target_paths), config.OUTPUT_DIM, config.OUTPUT_DIM, 3))

    for input_path, target_path in zip(input_paths, target_paths):

        input = cv2.imread(input_path)
        target = cv2.imread(target_path)

        input_writer.add([input], [-1])
        target_writer.add([target], [-1])

    input_writer.close()
    target_writer.close()

    # Remove temporary directory
    shutil.rmtree(config.LR_IMAGES)
    shutil.rmtree(config.HR_IMAGES)


if __name__ == '__main__':
    main()
