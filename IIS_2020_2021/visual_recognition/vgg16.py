import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm
from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras.api.keras.applications.vgg16 import VGG16, preprocess_input


def load_vgg16(fc):
    """ Creates VGG16 model.

    :param fc: fully connected layer as output layer if true
    :type fc: bool
    :return: instance of VGG16 keras model
    :rtype: keras.Model
    """
    base_model = VGG16(include_top=True, weights='imagenet', input_shape=(224, 224, 3))
    if fc:
        model = Model(inputs=base_model.input, outputs=base_model.get_layer(name='fc2').output)
    else:
        model = Model(inputs=base_model.input, outputs=base_model.get_layer(name='block5_pool').output)
    model.trainable = False
    return model


def create_features(image_id, model):
    """ Creates features with VGG16 model for given image.

    :param image_id: id of the image
    :type image_id: str
    :param model: VGG16 model
    :type model: keras.Model
    :return: features of the image
    :rtype: numpy.array
    """
    img = cv2.resize(cv2.imread(f'../visual genome/{image_id}.jpg'), (224, 224))
    features = model.predict(preprocess_input(np.expand_dims(img.astype(np.float32), axis=0)))
    return features[0]


def create_features_parallel(image_ids, model):
    """ Creates features with VGG16 model for given image.

    :param image_ids: ids of the images
    :type image_ids: list
    :param model: VGG16 model
    :type model: keras.Model
    :return: features of the image
    :rtype: numpy.array
    """
    input_f = []
    for image_id in image_ids:
        print("START", image_id)  # last printed DONE 2414570
        img = cv2.imread("./visual genome/" + str(image_id) + '.jpg')
        img = cv2.resize(img, (224, 224))
        input_f.append(img.astype(np.float32))

    features = model.predict(preprocess_input(np.array(input_f)))
    return features


def load_vgg16_features(image_id, fc):
    """ Loads VGG16 features for the image with given id. It assumes that the features are already created.

    :param image_id: id of the image
    :type image_id: str
    :param fc: use features from fully connected layer if true
    :type fc: bool
    :return: features of the image
    :rtype: numpy.array
    """
    if fc:
        with open(f'../dataset/features/vgg16/{image_id}.pkl', 'rb') as f:
            features = pickle.load(f)
    else:
        with open(f'../dataset/features/vgg16-conv/{image_id}.pkl', 'rb') as f:
            features = pickle.load(f)
    return features


def create_vgg16_features_parallel(image_ids, fc):
    """ Creates VGG16 features for images with given ids. Features are saved to a file

    :param image_ids: image ids
    :type image_ids: numpy.array
    :param fc: use features from fully connected layer if true
    :type fc: bool
    """
    vgg_16_model = load_vgg16(fc)
    images = []
    for i, image_id in zip(tqdm(list(range(len(image_ids)))), image_ids):
        if fc:
            features_path = f'../dataset/features/vgg16/{image_id}.pkl'
        else:
            features_path = f'../dataset/features/vgg16-conv/{image_id}.pkl'
        if not os.path.exists(features_path):
            images.append(image_id)
        if len(images) == 16 or i == len(image_ids) - 1:
            features = create_features_parallel(images, vgg_16_model)
            for im_id, feats in zip(images, features):
                if fc:
                    with open(f'../dataset/features/vgg16/{im_id}.pkl', 'wb') as f:
                        pickle.dump(feats, f)
                else:
                    with open(f'../dataset/features/vgg16-conv/{im_id}.pkl', 'wb') as f:
                        pickle.dump(feats, f)
            images = []


def create_vgg16_features(image_ids, fc):
    """ Creates VGG16 features for images with given ids. Features are saved to a file

    :param image_ids: image ids
    :type image_ids: numpy.array
    :param fc: use features from fully connected layer if true
    :type fc: bool
    """
    vgg_16_model = load_vgg16(fc)
    for _, image_id in zip(tqdm(list(range(len(image_ids)))), image_ids):
        if fc:
            if not os.path.exists(f'../dataset/features/vgg16/{image_id}.pkl'):
                features = create_features(image_id, vgg_16_model)
                with open(f'../dataset/features/vgg16/{image_id}.pkl', 'wb') as f:
                    pickle.dump(features, f)
        else:
            if not os.path.exists(f'../dataset/features/vgg16-conv/{image_id}.pkl'):
                features = create_features(image_id, vgg_16_model)
                with open(f'../dataset/features/vgg16-conv/{image_id}.pkl', 'wb') as f:
                    pickle.dump(features, f)


if __name__ == '__main__':
    vgg_16_model = load_vgg16(False)
    create_features('KITP-11-22560-g004',
                    vgg_16_model)
    # train_ids = load_image_ids('train')
    # create_vgg16_features_parallel(train_ids, False)
    # val_ids = load_image_ids('val')
    # create_vgg16_features_parallel(val_ids, False)
    # test_ids = load_image_ids('test')
    # create_vgg16_features_parallel(test_ids, False)
