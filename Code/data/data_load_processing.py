import cv2
import os
import numpy as np

def load_images_from_folder(folder, n_imgs=-1):
    images = []
    image_nums = []
    for filename in os.listdir(folder)[:n_imgs]:
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            image_nums.append(filename.strip('.png'))
    return images, image_nums

def label_preprocessing(labels):
    labels['img_number'] = labels['filename'].apply(lambda x: x.split('/')[-1].strip('.png'))
    label_array = labels[labels['img_number'].isin(fnames)]['latex'].values
    label_array = [f'\t{la}\n' for la in label_array]
    return label_array

def image_preprocessing_for_inception(images):
    images = np.array(images)
    images = 255 - images
    return images

def image_preprocessing(images):
    images = np.array(images)
    images = 255 - images
    images = tf.image.rgb_to_grayscale(images)
    images = images.numpy()
    return images
