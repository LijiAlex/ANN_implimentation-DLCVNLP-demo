import yaml
import time
import matplotlib.pyplot as plt
import pandas as pd
import os


def get_unique_filename(filename):
    unique_filename = time.strftime(f"%Y-%m-%d_%H%S%M_{filename}")
    return unique_filename


def read_config(config_path):
    with open(config_path, 'r') as config_file:
        content = yaml.safe_load(config_file)
    return content


def save_plot(history, plot_name, plot_dir, logger):
    unique_filename = get_unique_filename(plot_name)
    path_to_model = os.path.join(plot_dir, unique_filename)
    pd.DataFrame(history.history).plot(figsize=(10, 7))
    plt.grid(True)
    plt.savefig(path_to_model)
    logger.info("Plot saved")