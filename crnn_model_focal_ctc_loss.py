"""Keras implementation of CRNN."""

import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Activation, Conv2D, MaxPool2D, BatchNormalization, LSTM, GRU
from keras.layers.wrappers import Bidirectional
from keras.layers import Reshape, Permute, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Dropout

from keras import regularizers

import tensorflow as tf

def CRNN(input_shape, num_classes, prediction_only=False, gru=True, alpha=0.75, gamma=0.5):
    """CRNN architecture.
    
    # Arguments
        input_shape: Shape of the input image, (256, 32, 1).
        num_classes: Number of characters in alphabet, including CTC blank.
        
    # References
        https://arxiv.org/abs/1507.05717
    """
    #K.clear_session()
    
    act = LeakyReLU(alpha=0.05)
    #act = 'relu'
    
    x = image_input = Input(shape=input_shape, name='image_input')
    x = Conv2D(64, (3, 3), strides=(1, 1), activation=act, padding='same', name='conv1_1')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='pool1', padding='same')(x)
    x = Conv2D(128, (3, 3), strides=(1, 1), activation=act, padding='same', name='conv2_1')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='pool2', padding='same')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), activation=act, padding='same', name='conv3_1')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), activation=act, padding='same', name='conv3_2')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(1, 2), name='pool3', padding='same')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), activation=act, padding='same', name='conv4_1')(x)
    x = BatchNormalization(name='batchnorm1')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), activation=act, padding='same', name='conv5_1')(x)
    x = BatchNormalization(name='batchnorm2')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(1, 2), name='pool5', padding='valid')(x)
    x = Conv2D(512, (2, 2), strides=(1, 1), activation=act, padding='valid', name='conv6_1')(x)
    x = Reshape((-1,512))(x)
    if gru:
        x = Bidirectional(GRU(256,dropout=0.1,recurrent_dropout=0.1, return_sequences=True))(x)
        x = Bidirectional(GRU(256,dropout=0.1,recurrent_dropout=0.1, return_sequences=True))(x)
    else:
        x = Bidirectional(LSTM(256, return_sequences=True,dropout=0.1,recurrent_dropout=0.1, name='lstm1'))(x)
        x = Bidirectional(LSTM(256, return_sequences=True,dropout=0.1,recurrent_dropout=0.1, name='lstm2'))(x)
    
    x = Dense(num_classes,
              #kernel_regularizer=regularizers.l2(0.01),
              #activity_regularizer=regularizers.l1(0.01),
              name='dense1')(x)
    #x = Dropout(0.1)(x)
    x = y_pred = Activation('softmax', name='softmax')(x)
    
    model_pred = Model(image_input, x)
    
    if prediction_only:
        return model_pred

    max_string_len = int(y_pred.shape[1])

    def focal_ctc_lambda_func(args):
        labels, y_pred, input_length, label_length = args
        ctc_loss = K.ctc_batch_cost(labels, y_pred, input_length, label_length)
        p = tf.exp(-ctc_loss)
        focal_ctc_loss = alpha*tf.pow((1-p),gamma)*ctc_loss
        return focal_ctc_loss

    labels = Input(name='label_input', shape=[max_string_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    focal_ctc_loss = Lambda(focal_ctc_lambda_func, output_shape=(1,), 
                            name='focal_ctc_loss')([labels, y_pred, input_length, label_length])
    
    model_train = Model(inputs=[image_input, labels, input_length, label_length],
                        outputs=focal_ctc_loss)
    
    return model_train, model_pred
    
