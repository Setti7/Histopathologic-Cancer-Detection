import tensorflow as tf
from tensorflow.keras import  backend as K

# https://stackoverflow.com/a/51436745/7343992
def auc(y_true, y_pred):

    # argmax is necessary because model.predict_generator has a different output from model.predict
    # https://github.com/keras-team/keras/issues/3477#issuecomment-360022086
    y_pred = K.argmax(y_pred, axis=-1)

    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())

    return auc