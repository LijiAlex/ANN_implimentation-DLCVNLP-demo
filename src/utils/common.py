import yaml
import matplotlib.pyplot as plt
import pandas as pd
import os
import time


def get_unique_filename(filename, path_dir):
    unique_filename = time.strftime(f"%Y-%m-%d_%H%S%M_{filename}")
    path = os.path.join(path_dir, unique_filename)
    return path


def read_config(config_path):
    with open(config_path, 'r') as config_file:
        content = yaml.safe_load(config_file)
    return content


def save_plot(history, plot_name, plot_dir, logger):
    path_to_model = get_unique_filename(plot_name, plot_dir)
    pd.DataFrame(history.history).plot(figsize=(10, 7))
    plt.grid(True)
    plt.savefig(path_to_model)
    logger.info("Plot saved")
