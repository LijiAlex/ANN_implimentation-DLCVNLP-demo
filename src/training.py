import os
from utils.common import read_config, save_plot
from utils.data_mgmt import get_data
from utils.model import create_model, save_model
import argparse


def training(config_path):
    config = read_config(config_path)
    validation_datasize = config["params"]["validation_datasize"]
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data(validation_datasize)

    loss_function = config["params"]["loss_functions"]
    optimizer = config["params"]["optimizer"]
    metrics = config["params"]["metrics"]
    model = create_model(loss_function, optimizer, metrics)

    epochs = config["params"]["epochs"]
    validation = (X_val, y_val)
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=validation)

    artifacts_dir = config["artifacts"]["artifacts_dir"]
    model_dir = config["artifacts"]["model_dir"]
    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)
    model_name = config["artifacts"]["model_name"]
    save_model(model, model_name, model_dir_path)

    plots_dir = config["artifacts"]["plots_dir"]
    plot_dir_path = os.path.join(artifacts_dir, plots_dir)
    os.makedirs(plot_dir_path, exist_ok=True)
    plot_name = config["artifacts"]["plot_name"]
    save_plot(history, plot_name, plot_dir_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', "-c", default="config.yaml")
    parsed_args = args.parse_args()

    ROOT_DIR = os. getcwd()
    print("ROOT Path", ROOT_DIR)
    training(config_path=parsed_args.config)
