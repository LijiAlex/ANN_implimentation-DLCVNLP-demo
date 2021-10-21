# ANN_implimentation
An ANN implementation for MNIST dataset.

## Package description
* src.training [Creates, trains and saves the model]
* src.predict [Tests the model]
* src.utils.common [Utility methods]
* src.utils.data_mgmt [Data preparation]
* src.utils.model [Model functions]

## Folder Structure
1. artifacts
    * checkpoints [Model checkpoints]
    * model [Saved models]
    * plots [Saved plots]
2. logs
   * tensorboard_logs [tensorboard logs for each iteration]
   * general_logs
3. src
   * utils
     * __init__.py
     * common.py
     * data_mgmt.py
     * model.py
   * __init__.py
   * predict.py
   * training.py
