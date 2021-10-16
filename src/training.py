import os
from src.utils.common import read_config, save_plot
from src.utils.data_mgmt import get_data
from src.utils.model import create_model, save_model
import argparse
import logging


def training(config_path):
    try:
        config = read_config(config_path)
        artifacts_dir = config["artifacts"]["artifacts_dir"]

        logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
        log_dir = config["logs"]["logs_dir"]
        general_logs = config["logs"]["general_logs"]
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(filename=os.path.join(log_dir, general_logs), level=logging.INFO, format=logging_str)
        logger = logging.getLogger()
        logger.info("************* Begin Training *************")

        validation_datasize = config["params"]["validation_datasize"]
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data(validation_datasize, logger)

        loss_function = config["params"]["loss_functions"]
        optimizer = config["params"]["optimizer"]
        metrics = config["params"]["metrics"]
        model = create_model(loss_function, optimizer, metrics, logger)

        epochs = config["params"]["epochs"]
        validation = (X_val, y_val)
        logger.info(f"Start training, Epochs:{epochs}")
        history = model.fit(X_train, y_train, epochs=epochs, validation_data=validation)
        logger.info(history.history)
        logger.info("Training successful")

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

        logger.info("************* End Training *************\n\n\n")
    except Exception:
        logger.info("Exception occurred.")
        logger.info("************* End Training *************\n\n\n")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', "-c", default="config.yaml")
    parsed_args = args.parse_args()
    training(config_path=parsed_args.config)
