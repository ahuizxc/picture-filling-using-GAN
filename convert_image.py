import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

DATASET_NAME='180256'
DATASET_NAME_TRAIN = 'train'
DATASET_NAME_TEST = '180256_te'

IMAGE_SIZE_H = 180
IMAGE_SIZE_W = 256
IMAGE_CHANNEL = 1
def convert_to_bin():
  dataset_path_tr = 'data/' + DATASET_NAME_TRAIN + '/'
  path_pattern_tr = dataset_path_tr + '*.jpg'
  images_tr = np.array(io.ImageCollection(path_pattern_tr))
  
  # dataset_path_te = 'data/' + DATASET_NAME_TEST + '/'
  # path_pattern_te = dataset_path_te + '*.jpg'
  # images_te = np.array(io.ImageCollection(path_pattern_te))
  
  print(images_tr.shape)
  # print(images_te.shape)


  image_train = images_tr[:]
  # image_test = images_te[:]
  image_train.tofile('data/' + DATASET_NAME + '_train.bin')
  # image_test.tofile('data/' + DATASET_NAME + '_test.bin')

def display_bin():
  file_object = open('data/' + DATASET_NAME + '_train.bin', 'rb')
  # file_object = open('data/' + DATASET_NAME + '_test.bin', 'rb')
  images = np.fromfile(file_object, dtype=np.uint8)
  images = np.reshape(images, (-1, IMAGE_SIZE_H, IMAGE_SIZE_W, IMAGE_CHANNEL))
  print(images.shape)
  plt.figure('image')
  print(images[0].shape)
  if IMAGE_CHANNEL == 1:
    plt.imshow(images[2, :, :, 0], cmap='gray')
  elif IMAGE_CHANNEL == 3:
    plt.imshow(images[13])
  else:
    print('image channel not supported')
  plt.show()


def main():
  convert_to_bin()
  display_bin()

if __name__ == '__main__':
  main()
