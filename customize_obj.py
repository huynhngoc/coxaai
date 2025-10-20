from deoxys.customize import custom_architecture, custom_preprocessor, custom_loss
from deoxys.loaders.architecture import BaseModelLoader
from deoxys.data.preprocessor import BasePreprocessor
from deoxys.utils import deep_copy
from deoxys.model.losses import loss_from_config, Loss


from tensorflow.keras.applications import efficientnet, efficientnet_v2
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Model
import tensorflow as tf

import numpy as np


@custom_architecture
class EfficientNetModelLoader(BaseModelLoader):
    """
    Create a sequential network from list of layers
    """
    map_name = {
        'B0': efficientnet.EfficientNetB0,
        'B1': efficientnet.EfficientNetB1,
        'B2': efficientnet.EfficientNetB2,
        'B3': efficientnet.EfficientNetB3,
        'B4': efficientnet.EfficientNetB4,
        'B5': efficientnet.EfficientNetB5,
        'B6': efficientnet.EfficientNetB6,
        'B7': efficientnet.EfficientNetB7,
        'S': efficientnet_v2.EfficientNetV2S,
        'M': efficientnet_v2.EfficientNetV2M,
        'L': efficientnet_v2.EfficientNetV2L,
    }

    def __init__(self, architecture, input_params):
        self._input_params = deep_copy(input_params)
        self.options = architecture

    def load(self):
        """

        Returns
        -------
        tensorflow.keras.models.Model
            A neural network of sequential layers
            from the configured layer list.
        """
        num_class = self.options['num_class']
        pretrained = self.options['pretrained']
        activation_type = self.options.get('activation', 'auto')
        shape = self._input_params['shape']
        efficientNet = self.map_name[self.options['class_name']]
        if activation_type == 'auto':
            if num_class <= 2:
                num_class = 1
                activation = 'sigmoid'
            else:
                activation = 'softmax'
        else:
            activation = activation_type

        if pretrained:
            model = efficientNet(include_top=False, classes=num_class,
                                 classifier_activation=activation, input_shape=shape, pooling='avg')
            if self.options.get('freeze', None) is not None:
                if self.options['freeze'] == 'all':
                    for layer in model.layers:
                        layer.trainable = False
                elif isinstance(self.options['freeze'], int):
                    # Freeze all layers except the last `freeze` layers
                    for layer in model.layers[:-self.options['freeze']]:
                        layer.trainable = False

            dropout_out = Dropout(0.3)(model.output)
            pred = Dense(num_class, activation=activation)(dropout_out)
            model = Model(model.inputs, pred)
        else:
            model = efficientNet(weights=None, include_top=True, classes=num_class,
                                 classifier_activation=activation, input_shape=shape)

        return model


@custom_preprocessor
class PretrainedEfficientNet(BasePreprocessor):
    def transform(self, images, targets):
        # efficientNet requires input between [0-255]
        images = images * 255
        # pretrain require 3 channel
        new_images = np.concatenate([images, images, images], axis=-1)

        return new_images, targets


@custom_preprocessor
class OneHot(BasePreprocessor):
    def __init__(self, num_class=2):
        if num_class <= 2:
            num_class = 1
        self.num_class = num_class

    def transform(self, images, targets):
        # labels to one-hot encode
        new_targets = np.zeros((len(targets), self.num_class))
        if self.num_class==1:
            new_targets[..., 0] = targets
        else:
            for i in range(self.num_class):
                new_targets[..., i][targets == i] = 1

        return images, new_targets


@custom_loss
class FusedLoss(Loss):
    """Used to sum two or more loss functions.
    """

    def __init__(
            self, loss_configs, loss_weights=None,
            reduction="auto", name="fused_loss"):
        super().__init__(reduction, name)
        self.losses = [loss_from_config(loss_config)
                       for loss_config in loss_configs]

        if loss_weights is None:
            loss_weights = [1] * len(self.losses)
        self.loss_weights = loss_weights

    def call(self, target, prediction):
        loss = None
        for loss_class, loss_weight in zip(self.losses, self.loss_weights):
            if loss is None:
                loss = loss_weight * loss_class(target, prediction)
            else:
                loss += loss_weight * loss_class(target, prediction)

        return loss


@custom_loss
class DiffPenalty(Loss):
    """Used to add a penalty to the loss based on the difference between
    the prediction and target.
    """

    def __init__(self, reduction="auto", name="diff_penalty"):
        super().__init__(reduction, name)

    def call(self, target, prediction):
        # Predicted class (argmax)
        y_pred_class = tf.argmax(prediction, axis=-1)

        # True class
        y_true_class = tf.argmax(target, axis=-1)
        class_diff = tf.cast(tf.abs(y_pred_class - y_true_class), prediction.dtype)

        return class_diff * class_diff
