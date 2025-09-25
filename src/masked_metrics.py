"""
This script provides masked metrics for proper accuracy, auc, recall, precision, F1 analysis.
Auth: Florian Zwicker
"""
import tensorflow as tf
from tensorflow.keras import backend as K
import keras

@tf.function
def masked_accuracy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
    if tf.rank(mask) + 1 == tf.rank(y_pred):
        mask = tf.expand_dims(mask, axis=-1)
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    correct = tf.cast(tf.equal(y_true, y_pred_bin), tf.float32)
    return tf.reduce_sum(correct * mask) / tf.reduce_sum(mask + K.epsilon())

@tf.function
def masked_precision(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Immer Mask für denselben Rank bauen
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)  # [batch, seq_len, 1]

    # Falls y_true z.B. nur [batch, seq_len] ist, zwingen:
    if y_true.shape.rank < y_pred.shape.rank:
        y_true = tf.expand_dims(y_true, axis=-1)
        mask = tf.expand_dims(mask, axis=-1)

    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)

    true_positives = tf.reduce_sum(y_pred_bin * y_true * mask)
    predicted_positives = tf.reduce_sum(y_pred_bin * mask)

    return true_positives / (predicted_positives + tf.keras.backend.epsilon())


@tf.function
def masked_recall(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)


    if y_true.shape.rank < y_pred.shape.rank:
        y_true = tf.expand_dims(y_true, axis=-1)
        mask = tf.expand_dims(mask, axis=-1)

    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)

    true_positives = tf.reduce_sum(y_pred_bin * y_true * mask)
    possible_positives = tf.reduce_sum(y_true * mask)

    return true_positives / (possible_positives + tf.keras.backend.epsilon())


@tf.function
def masked_auc(y_true, y_pred):
    if tf.rank(y_true) > 2:
        y_true = tf.squeeze(y_true, axis=-1)
        y_pred = tf.squeeze(y_pred, axis=-1)

    # Maske bestimmen
    mask = tf.not_equal(y_true, -1)
    if tf.rank(mask) + 1 == tf.rank(y_pred):
        mask = tf.expand_dims(mask, axis=-1)

    # Nur echte Werte
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)

    # Berechne AUC nur auf gültigen Daten
    auc = tf.keras.metrics.AUC()
    auc.update_state(y_true_masked, y_pred_masked)
    return auc.result()

class MaskedAUC(tf.keras.metrics.AUC):
    def __init__(self, name="masked_auc", **kwargs):
        super(MaskedAUC, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Maske anwenden (ignoriere Padding-Tokens -1)
        mask = tf.not_equal(y_true, -1)

        # Nur gültige Werte beibehalten
        y_true_masked = tf.boolean_mask(y_true, mask)
        y_pred_masked = tf.boolean_mask(y_pred, mask)

        # Berechne AUC mit den validen (nicht gepaddeten) Werten
        super(MaskedAUC, self).update_state(y_true_masked, y_pred_masked, sample_weight=sample_weight)

    def result(self):
        return super(MaskedAUC, self).result()

@tf.function
def masked_f1_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # static mask
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)

    # shape fix
    if y_true.shape.rank < y_pred.shape.rank:
        y_true = tf.expand_dims(y_true, axis=-1)
        mask = tf.expand_dims(mask, axis=-1)

    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)

    tp = tf.reduce_sum(y_pred_bin * y_true * mask)
    predicted_positives = tf.reduce_sum(y_pred_bin * mask)
    possible_positives = tf.reduce_sum(y_true * mask)

    precision = tp / (predicted_positives + tf.keras.backend.epsilon())
    recall = tp / (possible_positives + tf.keras.backend.epsilon())

    return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())


@tf.function
def masked_mcc(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Mask: True labels that are not -1
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)

    # If needed, expand dims for broadcasting
    if y_true.shape.rank < y_pred.shape.rank:
        y_true = tf.expand_dims(y_true, axis=-1)
        mask = tf.expand_dims(mask, axis=-1)

    # Binarize predictions
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)

    # True Positives, False Positives, True Negatives, False Negatives (masked)
    tp = tf.reduce_sum(y_pred_bin * y_true * mask)
    tn = tf.reduce_sum((1 - y_pred_bin) * (1 - y_true) * mask)
    fp = tf.reduce_sum(y_pred_bin * (1 - y_true) * mask)
    fn = tf.reduce_sum((1 - y_pred_bin) * y_true * mask)

    numerator = tp * tn - fp * fn
    denominator = tf.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )

    return numerator / (denominator + tf.keras.backend.epsilon())


# use these for evaluation after loading a model. Have to be implemented for the whole pipeline to work
masked_mcc_metric = tf.keras.metrics.MeanMetricWrapper(masked_mcc, name="masked_mcc")
masked_precision_metric = tf.keras.metrics.MeanMetricWrapper(masked_precision, name="masked_precision")
masked_recall_metric = tf.keras.metrics.MeanMetricWrapper(masked_recall, name="masked_recall")
masked_f1_score_metric = tf.keras.metrics.MeanMetricWrapper(masked_f1_score, name="masked_f1_score")
