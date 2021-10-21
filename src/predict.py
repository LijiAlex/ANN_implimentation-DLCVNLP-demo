from src.utils.data_mgmt import get_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import logging

validation_datasize = 5000
logger = logging.getLogger()
(X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data(validation_datasize, logger)
X_new = X_test[:3]
model = tf.keras.models.load_model("artifacts/model/2021-10-21_061843_model.h5")
y_prob = model.predict(X_new)
y_prob.round(3)
Y_pred = np.argmax(y_prob, axis=-1)  # axis = -1 , give you output for each array/row
for img_array, pred, actual in zip(X_new, Y_pred, y_test[:3]):
    plt.imshow(img_array, cmap="binary")
    plt.title(f"predicted: {pred}, Actual: {actual}")
    plt.axis("off")
    plt.show()
    print("_____"*20)
