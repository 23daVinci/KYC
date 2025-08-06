import tensorflow as tf
import io
import os
from PIL import Image
import pandas as pd

from utils import load_config 


############## Global Variables ##############

CONFIG = load_config()

##############################################


class DataSerializer:
    """
    Class to serialize image pairs into TFRecord format for training a signature matching model.
    """
    def __init__(self):
        pass


    def load_and_encod_images(self, img_path: str) -> tf.image.encode_png:
        """
        Loads and image from its path, converts it to grayscale, and standardises it.

        Args:
            img_path (str): Disk path of the image

        Returns:
            Byte string representation of the image
        """
        image = Image.open(img_path).convert('L')  # convert to grayscale
        image = image.resize((220, 155))       # standardize size
        # Convert to numpy and add channel dimension: [H, W] → [H, W, 1]
        image_array = tf.convert_to_tensor(image, dtype=tf.uint8)
        image_array = tf.expand_dims(image_array, axis=-1)  # Now shape is (155, 220, 1)
        return tf.image.encode_png(image_array)
    

    def create_example(self, img1_bytes: tf.image.encode_png, img2_bytes: tf.image.encode_png, label) -> tf.train.Example:
        """
        Wraps an image pair into tf.train.Example.

        Args:
            img1_bytes (tf.image.encode_png): Byte string representation of the first image
            img2_bytes (tf.image.encode_png): Byte string representation of the second image
            label (int): Pair label

        Returns:
            tf.train.Example of the image pair.
        """
        feature = {
            'image1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img1_bytes.numpy()])),
            'image2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img2_bytes.numpy()])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))
    

    def serialize(self, set: str) -> None:
        """ 
        Serializes image pairs into a TFRecord file.
        
        Args:
            set (str): The dataset type, can be 'train', 'val', or 'test'.

        Returns:
            None: Writes the serialized data to a TFRecord file.
        """
        # Get image pairs
        image_pairs = self._get_image_pairs(set)

        with tf.io.TFRecordWriter('../data/train_pairs.tfrecord') as writer:
            try:
                for img1_path, img2_path, label in image_pairs:
                    img1_bytes = self.load_and_encod_images(img1_path)
                    img2_bytes = self.load_and_encod_images(img2_path)
                    example = self.create_example(img1_bytes, img2_bytes, label)
                    writer.write(example.SerializeToString())
            except Exception as e:
                print(f"Error processing image pair {img1_path}, {img2_path}: {e}")
                raise e
            else:
                print(f"Serialized {len(image_pairs)} image pairs to TFRecord for {set} set.")


    def _get_image_pairs(self, set: str) -> list[tuple]:
        """
        Reads the CSV file containing image pairs and their labels, and returns a list of tuples.
        
        Args:
            set (str): The dataset type, can be 'train', 'val', or 'test'.

        Returns:
            list(tuple): A list of tuples where each tuple contains the paths of two images and their label.
        """
        if set == 'train':
            CSV_PATH = CONFIG['data']['train_pairs']
            IMG_DIR = CONFIG['data']['train_dir']
        elif set == 'val':
            CSV_PATH = CONFIG['data']['val_pairs']   
            IMG_DIR = CONFIG['data']['val_dir']
        else:
            CSV_PATH = CONFIG['data']['test_pairs']
            IMG_DIR = CONFIG['data']['test_dir']

        img_pairs_df = pd.read_csv(CSV_PATH, header=None, names=['img1', 'img2', 'label'])

        # Append full path to each image and convert to tuple list
        image_pair_list = [
                            (
                                os.path.normpath(os.path.join(IMG_DIR, row.img1)),
                                os.path.normpath(os.path.join(IMG_DIR, row.img2)),
                                row.label
                            )
                            for row in img_pairs_df.itertuples(index=False)
                          ]

        return image_pair_list
        

           


if __name__ == "__main__":
    serializer = DataSerializer()
    serializer.serialize('train')