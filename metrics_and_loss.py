import tensorflow as tf
from tensorflow import keras


@keras.saving.register_keras_serializable(package="CCEplusJACCARD_Loss")
class CombinedLoss(tf.keras.losses.Loss):
    """
    Custom loss function combining Jaccard Distance and Categorical Cross-Entropy
    used for image segmentation. usefull with imbalance classes
    """
    def __init__(self, smooth=100, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.smooth = smooth
        self.alpha = alpha  # Scalar to weight the losses
        self.cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        
    def call(self, y_true, y_pred):
        # Jaccard Loss
        intersection = tf.reduce_sum(y_true * y_pred, axis=(1,2))
        sum_ = tf.reduce_sum(y_true + y_pred, axis=(1,2))
        jac = (intersection + self.smooth) / (sum_ - intersection + self.smooth)
        jaccard_loss = (1 - jac) * self.smooth
        jaccard_loss = tf.reduce_mean(jaccard_loss)
        
        # Categorical Cross Entropy
        cce_loss = self.cce(y_true, y_pred)
        
        # Combining losses
        return self.alpha * jaccard_loss + (1 - self.alpha) * cce_loss
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'smooth': self.smooth, 'alpha': self.alpha}

@keras.saving.register_keras_serializable(package="CustomMetrics")
class IoUMetric(tf.keras.metrics.Metric):
    """
    Métrique personnalisée pour calculer l'IoU (Intersection over Union)
    adaptée pour la segmentation d'images.
    """
    def __init__(self, name='iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.iou_sum = self.add_weight(name='iou_sum', initializer='zeros')
        self.total_samples = self.add_weight(name='total_samples', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convertir les prédictions en binaire (0/1) si nécessaire
        y_pred = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
        
        # Calculer intersection et union
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        union = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2]) - intersection
        
        # Éviter division par zéro
        iou = tf.where(tf.equal(union, 0), 1.0, intersection / union)
        
        batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
        self.iou_sum.assign_add(tf.reduce_sum(iou))
        self.total_samples.assign_add(batch_size)
        
    def result(self):
        return self.iou_sum / self.total_samples
        
    def reset_state(self):
        self.iou_sum.assign(0.0)
        self.total_samples.assign(0.0)
    
    def get_config(self):
        base_config = super().get_config()
        return base_config

