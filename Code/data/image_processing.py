import tensorflow as tf
import numpy as np

def image_inception(X_train, X_test):
    X_train = tf.image.resize_with_pad(X_train, 299, 299)
    X_test = tf.image.resize_with_pad(y_train, 299, 299)

    X_train = tf.keras.applications.inception_v3.preprocess_input(X_train)
    X_test = tf.keras.applications.inception_v3.preprocess_input(X_test)

    return X_train, X_test

def making_batch_features(X, y, batch_size=50, filename):
    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    image_dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size)
    

    for img, label in image_dataset:
        batch_features = image_features_extract_model(img)
        batch_features = tf.reshape(batch_features,
                              (batch_features.shape[0], -1, batch_features.shape[3]))
  
        p = f'../{filename}/batch_features{c}'
        np.save(p, batch_features)

def loading_batch_features(total_batch_features, filename):
    img_load = np.load('../{filename}/batch_features1.npy')
    for i in range(2,total_batch_features):
        img_add = np.load(f'../{filename}/batch_features{i}.npy')
        img_load = np.concatenate((img_load, img_add))

    return img_load
