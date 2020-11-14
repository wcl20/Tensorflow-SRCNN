import argparse
import cv2
import numpy as np
import os
from config import config
from PIL import Image
from tensorflow.keras.models import load_model

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Path to input image")
    args = parser.parse_args()

    image = cv2.imread(args.input)
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

    # Upscale image using bicubic interpolation
    high_width = int(low_width * (config.SCALE / 1.))
    high_height = int(low_height * (config.SCALE / 1.))
    hr_image = Image.fromarray(lr_image).resize((high_width, high_height), resample=Image.BICUBIC)
    hr_image = np.array(hr_image)

    # Save bicubic result
    cv2.imwrite(os.path.sep.join([config.OUTPUT_DIR, "bicubic.png"]), hr_image)

    # Create output
    output = np.zeros(hr_image.shape)
    height, width = output.shape[:2]

    # Load SR model
    model = load_model(config.MODEL_PATH)

    for y in range(0, height - config.INPUT_DIM + 1, config.OUTPUT_DIM):
        for x in range(0, width - config.INPUT_DIM + 1, config.OUTPUT_DIM):
            # Get region of interest
            roi = hr_image[y:y+config.INPUT_DIM, x:x+config.INPUT_DIM].astype("float32")
            # Apply super resolution
            pred = model.predict(np.expand_dims(roi, axis=0))
            pred = pred.reshape((config.OUTPUT_DIM, config.OUTPUT_DIM, 3))
            # Patch image
            output[y+config.PAD : y+config.PAD+config.OUTPUT_DIM, x+config.PAD : x+config.PAD+config.OUTPUT_DIM] = pred

    output = output[config.PAD : height - (height % config.INPUT_DIM + config.PAD),
                    config.PAD : width - (width % config.INPUT_DIM + config.PAD)]
    ouput = np.clip(output, 0, 255).astype("uint8")
    cv2.imwrite(os.path.sep.join([config.OUTPUT_DIR, "sr.png"]), output)


if __name__ == '__main__':
    main()
