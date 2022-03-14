import tensorflow as tf
import datetime


def learning_rate_schedule():
    return tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))


def create_tensorboard_callback(dir_name, experiment_name):
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir
     )
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback


# Example of a custom callback
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        # Check accuracy
        if logs.get('accuracy') > 0.95 :
            # Stop if threshold is met
            print("\nAccuracy is higher than 0.95 so cancelling training!")
            self.model.stop_training = True