import tensorflow as tf

@tf.function()
def custom_loss(y_true, y_pred):
    tf.print("hem entrat a les custom")
    tf.print("y_true =",y_true)
    tf.print("y_pred =",y_pred)

    