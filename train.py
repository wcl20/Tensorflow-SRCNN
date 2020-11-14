import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from config import config
from core.io import HDF5Reader
from core.nn import SRCNN
from tensorflow.keras.optimizers import Adam

def main():

    # Low resolution images
    input_gen = HDF5Reader(config.LR_HDF5, config.BATCH_SIZE)
    # High resolution images
    target_gen = HDF5Reader(config.HR_HDF5, config.BATCH_SIZE)
    # Define generator
    def generator(input_gen, target_gen):
        while True:
            input = next(input_gen)[0]
            target = next(target_gen)[0]
            yield input, target

    print("[INFO] Building model ...")
    model = SRCNN.build(config.INPUT_DIM, config.INPUT_DIM, 3)
    optimizer = Adam(lr=0.001, decay=0.001 / config.EPOCHS)
    model.compile(loss="mse", optimizer=optimizer)

    H = model.fit(
        generator(input_gen.generator(), target_gen.generator()),
        epochs=config.EPOCHS,
        steps_per_epoch=input_gen.num_images // config.BATCH_SIZE
    )

    input_gen.close()
    target_gen.close()

    print("[INFO] Saving model ...")
    model.save(config.MODEL_PATH, overwrite=True)

    # Plot Graph
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, config.EPOCHS), H.history["loss"], label="loss")
    plt.title("MSE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(config.PLOT_PATH)


if __name__ == '__main__':
    main()
