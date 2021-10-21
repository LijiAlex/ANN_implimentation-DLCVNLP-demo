import os
from src.utils.common import read_config, save_plot, get_unique_filename
from src.utils.data_mgmt import get_data
from src.utils.model import create_model, save_model
import argparse
import logging
import tensorflow as tf
import numpy as np


def training(config_path):
    logger = logging.getLogger()
    try:
        config = read_config(config_path)
        artifacts_dir = config["artifacts"]["artifacts_dir"]
        log_dir = config["logs"]["logs_dir"]
        general_logs = config["logs"]["general_logs"]
        tensorboard_logs = config["logs"]["tensorboard_logs"]
        os.makedirs(os.path.join(log_dir, tensorboard_logs), exist_ok=True)

        tensorboard_log_file = get_unique_filename("tensorboard", os.path.join(log_dir, tensorboard_logs))
        log_writer = tf.summary.create_file_writer(tensorboard_log_file)

        logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
        logging.basicConfig(filename=os.path.join(log_dir, general_logs), level=logging.INFO, format=logging_str)
        logger.info("************* Begin Training *************")

        validation_datasize = config["params"]["validation_datasize"]
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data(validation_datasize, logger)
        with log_writer.as_default():
            images = np.reshape(X_train[10:30], (-1, 28, 28, 1))  # take 20 images of 28 X 28 shape
            tf.summary.image("20 handwritten digit samples", images, max_outputs=25, step=0)

        loss_function = config["params"]["loss_functions"]
        optimizer = config["params"]["optimizer"]
        metrics = config["params"]["metrics"]
        model = create_model(loss_function, optimizer, metrics, logger)

        # call backs
        tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_file)
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ckpt_dir = config["artifacts"]["checkpoint_dir"]
        ckpt_model_dir_path = os.path.join(artifacts_dir, ckpt_dir)
        os.makedirs(ckpt_model_dir_path, exist_ok=True)
        ckpt_path = os.path.join(ckpt_model_dir_path, "model_ckpt.h5")
        checkpointing_cb = tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True)
        callbacks_list = [tensorboard_cb, early_stopping_cb, checkpointing_cb]

        epochs = config["params"]["epochs"]
        validation = (X_val, y_val)
        logger.info(f"Start training, Epochs:{epochs}")
        history = model.fit(X_train, y_train, epochs=epochs, validation_data=validation, callbacks=callbacks_list)
        logger.info(history.history)
        logger.info("Training successful")

        model.evaluate(X_test, y_test)
        logger.info("Evaluation successful")

        model_dir = config["artifacts"]["model_dir"]
        model_dir_path = os.path.join(artifacts_dir, model_dir)
        os.makedirs(model_dir_path, exist_ok=True)
        model_name = config["artifacts"]["model_name"]
        save_model(model, model_name, model_dir_path, logger)

        plots_dir = config["artifacts"]["plots_dir"]
        plot_dir_path = os.path.join(artifacts_dir, plots_dir)
        os.makedirs(plot_dir_path, exist_ok=True)
        plot_name = config["artifacts"]["plot_name"]
        save_plot(history, plot_name, plot_dir_path, logger)
    except Exception as exception:
        logger.info("Exception occurred.")
        logger.error(exception, exc_info=True)
    finally:
        logger.info("************* End Training *************\n\n\n")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', "-c", default="config.yaml")
    parsed_args = args.parse_args()
    training(config_path=parsed_args.config)
