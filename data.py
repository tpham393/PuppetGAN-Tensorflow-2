import numpy as np
import tensorflow as tf
import tf2lib as tl
from PIL import Image
import tensorflow.keras as keras


# def split_B(B_tuple_img):
#     # need the following line assuming B_tuple_img is a pathname
#     img = Image.open(B_tuple_img)

#     b2_box = (0, 0, 32, 32)
#     b2 = img.crop(b2_box)

#     b1_box = (0, 32, 32, 64)
#     b1 = img.crop(b1_box)

#     b3_box = (0, 64, 32, 96)
#     b3 = img.crop(b3_box)

#     return b1, b2, b3

def split_B(B_tuple_img_tensor):
    b2 = tf.image.crop_to_bounding_box(B_tuple_img_tensor, 0, 0, 32, 32)
    b1 = tf.image.crop_to_bounding_box(B_tuple_img_tensor, 32, 0, 32, 32)
    b3 = tf.image.crop_to_bounding_box(B_tuple_img_tensor, 64, 0, 32, 32)

    return b1, b2, b3

def uncompress(embedding):
    embedding = keras.layers.Dense(tf.keras.backend.prod(embedding.shape))(embedding)
    embedding = tf.reshape(embedding, (1, 32, 32, 3)) # unflatten
    print(embedding.shape)
    
    return embedding

def make_dataset(img_paths, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=True, repeat=1):
    if training:
        @tf.function
        def _map_fn(img):  # preprocessing
            # img = tf.image.random_flip_left_right(img)
            # img = tf.image.resize(img, [load_size, load_size])
            # img = tf.image.random_crop(img, [crop_size, crop_size, tf.shape(img)[-1]])
            # img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
            # img = img * 2 - 1
            img = tf.image.convert_image_dtype(img, dtype=tf.float32, saturate=False) / 255.0
            return img
    else:
        @tf.function
        def _map_fn(img):  # preprocessing
            # img = tf.image.resize(img, [crop_size, crop_size])  # or img = tf.image.resize(img, [load_size, load_size]); img = tl.center_crop(img, crop_size)
            # img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
            # img = img * 2 - 1
            img = tf.image.convert_image_dtype(img, dtype=tf.float32, saturate=False) / 255.0
            return img

    return tl.disk_image_batch_dataset(img_paths,
                                       batch_size,
                                       drop_remainder=drop_remainder,
                                       map_fn=_map_fn,
                                       shuffle=shuffle,
                                       repeat=repeat)


def make_zip_dataset(A_img_paths, B_img_paths, batch_size, load_size, crop_size, training, shuffle=True, repeat=False):
    # zip two datasets aligned by the longer one
    if repeat:
        A_repeat = B_repeat = None  # cycle both
    else:
        if len(A_img_paths) >= len(B_img_paths):
            A_repeat = 1
            B_repeat = None  # cycle the shorter one
        else:
            A_repeat = None  # cycle the shorter one
            B_repeat = 1

    A_dataset = make_dataset(A_img_paths, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=shuffle, repeat=A_repeat)
    B_dataset = make_dataset(B_img_paths, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=shuffle, repeat=B_repeat)

    A_B_dataset = tf.data.Dataset.zip((A_dataset, B_dataset))
    len_dataset = max(len(A_img_paths), len(B_img_paths)) // batch_size

    return A_B_dataset, len_dataset


class ItemPool:

    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.items = []

    def __call__(self, in_items):
        # `in_items` should be a batch tensor

        if self.pool_size == 0:
            return in_items

        out_items = []
        for in_item in in_items:
            if len(self.items) < self.pool_size:
                self.items.append(in_item)
                out_items.append(in_item)
            else:
                if np.random.rand() > 0.5:
                    idx = np.random.randint(0, len(self.items))
                    out_item, self.items[idx] = self.items[idx], in_item
                    out_items.append(out_item)
                else:
                    out_items.append(in_item)
        return tf.stack(out_items, axis=0)
