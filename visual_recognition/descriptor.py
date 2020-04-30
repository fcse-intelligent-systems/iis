import os
import cv2
import numpy as np


def create_descriptor_features(image_files):
    """Create features for images with SIFT descriptor

    :param image_files: list of images to be processed
    :type image_files: list(str)
    :return: numpy array of the created features
    :rtype: np.array
    """
    trainer = cv2.BOWKMeansTrainer(clusterCount=100)
    sift = cv2.xfeatures2d.SIFT_create()
    matcher = cv2.FlannBasedMatcher_create()
    bow_extractor = cv2.BOWImgDescriptorExtractor(sift, matcher)

    print('Creating dictionary')
    if os.path.exists('data/dictionary.npy'):
        dictionary = np.load('data/dictionary.npy')
    else:
        for filename in image_files:
            file = f'data/images/{filename.lower()}'
            img = cv2.imread(file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            key_points, desc_obj = sift.detectAndCompute(img, mask=None)
            trainer.add(desc_obj)

        dictionary = trainer.cluster()
        np.save('data/dictionary.npy', dictionary)

    bow_extractor.setVocabulary(dictionary)

    feature_data = np.zeros(shape=(len(image_files), dictionary.shape[0]),
                            dtype=np.float32)

    print('Extract features')
    for i, filename in zip(range(len(image_files)), image_files):
        file = f'data/images/{filename.lower()}'
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        points = sift.detect(img)
        feature_data[i] = bow_extractor.compute(img, points)

    return feature_data
