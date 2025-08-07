import os
import tensorflow as tf
from PIL import Image
import numpy as np
import tempfile

from scripts.preprocess import DataSerializer  


tf.get_logger().setLevel('ERROR')


def test_load_and_encode_image():
    # Create a temporary grayscale image
    img_array = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
    img = Image.fromarray(img_array, mode='L')

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_path = f"artifacts/{tmp.name}"
        img.save(tmp.name)

    serializer = DataSerializer()
    encoded = serializer.load_and_encode_image(temp_path)

    # Check if returned object is a Tensor
    assert isinstance(encoded, tf.Tensor)

    # Check if dtype is string (byte string)
    assert encoded.dtype == tf.string

    # Decode to check size and shape
    decoded_img = tf.image.decode_png(encoded, channels=1)
    assert decoded_img.shape == (155, 220, 1)  # Height, Width, Channels

    # Cleanup
    os.remove(temp_path)
