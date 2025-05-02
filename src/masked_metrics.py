"""
This script provides masked metrics for proper accuracy, auc, recall, precision, F1 analysis.
Auth: Florian Zwicker
"""
import tensorflow as tf
from tensorflow.keras import backend as K
import keras


def masked_accuracy(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    correct = tf.cast(tf.equal(y_true, y_pred_bin), tf.float32)
    return tf.reduce_sum(correct * mask) / tf.reduce_sum(mask + K.epsilon())

def masked_precision(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    true_positives = tf.reduce_sum(tf.cast(y_pred_bin * y_true, tf.float32) * mask)
    predicted_positives = tf.reduce_sum(y_pred_bin * mask)
    return true_positives / (predicted_positives + K.epsilon())

def masked_recall(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    true_positives = tf.reduce_sum(tf.cast(y_pred_bin * y_true, tf.float32) * mask)
    possible_positives = tf.reduce_sum(y_true * mask)
    return true_positives / (possible_positives + K.epsilon())

def masked_auc(y_true, y_pred):
    y_true = tf.squeeze(y_true, axis=-1)
    y_pred = tf.squeeze(y_pred, axis=-1)

    # Maske bestimmen
    mask = tf.not_equal(y_true, -1)

    # Nur echte Werte
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)

    # Berechne AUC nur auf gültigen Daten
    auc = tf.keras.metrics.AUC()
    auc.update_state(y_true_masked, y_pred_masked)
    return auc.result()