import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tf2lib as tl
import tqdm

import data
import module

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--experiment_dir')
py.arg('--batch_size', type=int, default=32)
test_args = py.args()
args = py.args_from_yaml(py.join(test_args.experiment_dir, 'settings.yml'))
args.__dict__.update(test_args.__dict__)


# ==============================================================================
# =                                    test                                    =
# ==============================================================================

# data
A_img_paths_test = py.glob(py.join(args.datasets_dir, 'real-digits', 'test'), '*.png')
B_img_paths_test = py.glob(py.join(args.datasets_dir, 'syn-digits', 'test'), '*.png')

A_B_dataset_test, _ = data.make_zip_dataset(A_img_paths_test, B_img_paths_test, args.batch_size, args.img_size, args.img_size, training=False, repeat=True)

# A_dataset_test = data.make_dataset(A_img_paths_test, args.batch_size, args.load_size, args.crop_size,
#                                    training=False, drop_remainder=False, shuffle=False, repeat=1)
# B_dataset_test = data.make_dataset(B_img_paths_test, args.batch_size, args.load_size, args.crop_size,
#                                    training=False, drop_remainder=False, shuffle=False, repeat=1)

# model
full_embed = module.encoder(input_shape=(args.img_size, args.img_size, 3))

decode_A = module.decoder(output_shape=(args.img_size, args.img_size, 3))
decode_B = module.decoder(output_shape=(args.img_size, args.img_size, 3))

# resotre
tl.Checkpoint(dict(full_embed=full_embed, decode_A=decode_A, decode_B=decode_B), py.join(args.experiment_dir, 'checkpoints')).restore()


@tf.function
def sample_to_A(A, B):
    b1, b2, b3 = data.split_B(B)

    attr_emb_A = tf.slice(full_embed(A, training=False), [0,0], [1,64])
    rest_emb_A = tf.slice(full_embed(A, training=False), [0,64], [1,64])

    attr_emb_b1 = tf.slice(full_embed(b1, training=False), [0,0], [1,64])
    rest_emb_b1 = tf.slice(full_embed(b1, training=False), [0,64], [1,64])
    attr_emb_b2 = tf.slice(full_embed(b2, training=False), [0,0], [1,64])
    rest_emb_b2 = tf.slice(full_embed(b2, training=False), [0,64], [1,64])
    attr_emb_b3 = tf.slice(full_embed(b3, training=False), [0,0], [1,64])
    rest_emb_b3 = tf.slice(full_embed(b3, training=False), [0,64], [1,64])

    Ab1 = decode_A(tf.reshape(tf.concat([attr_emb_b1, rest_emb_A],1), shape=[1,1,128]), training=False)
    Ab2 = decode_A(tf.reshape(tf.concat([attr_emb_b2, rest_emb_A],1), shape=[1,1,128]), training=False)
    Ab3 = decode_A(tf.reshape(tf.concat([attr_emb_b3, rest_emb_A],1), shape=[1,1,128]), training=False)
    return Ab1, Ab2, Ab3


@tf.function
def sample_to_B(A, B):
    b1, b2, b3 = data.split_B(B)

    attr_emb_A = tf.slice(full_embed(A, training=False), [0,0], [1,64])
    rest_emb_A = tf.slice(full_embed(A, training=False), [0,64], [1,64])

    attr_emb_b1 = tf.slice(full_embed(b1, training=False), [0,0], [1,64])
    rest_emb_b1 = tf.slice(full_embed(b1, training=False), [0,64], [1,64])
    attr_emb_b2 = tf.slice(full_embed(b2, training=False), [0,0], [1,64])
    rest_emb_b2 = tf.slice(full_embed(b2, training=False), [0,64], [1,64])
    attr_emb_b3 = tf.slice(full_embed(b3, training=False), [0,0], [1,64])
    rest_emb_b3 = tf.slice(full_embed(b3, training=False), [0,64], [1,64])

    b1A = decode_B(tf.reshape(tf.concat([attr_emb_A, rest_emb_b1],1), shape=[1,1,128]), training=False)
    b2A = decode_B(tf.reshape(tf.concat([attr_emb_A, rest_emb_b2],1), shape=[1,1,128]), training=False)
    b3A = decode_B(tf.reshape(tf.concat([attr_emb_A, rest_emb_b3],1), shape=[1,1,128]), training=False)
    b1b2 = decode_B(tf.reshape(tf.concat([attr_emb_b1, rest_emb_b2],1), shape=[1,1,128]), training=False)
    b2b3 = decode_B(tf.reshape(tf.concat([attr_emb_b2, rest_emb_b3],1), shape=[1,1,128]), training=False)
    return b1A, b2A, b3A, b1b2, b2b3


# run
test_iter = iter(A_B_dataset_test)
save_dir = py.join(args.experiment_dir, 'samples_testing')
py.mkdir(save_dir)

for A, B in tqdm.tqdm(A_B_dataset_test):
    A, B = next(test_iter)
    Ab1, Ab2, Ab3 = sample_to_A(A, B)
    b1A, b2A, b3A, b1b2, b2b3 = sample_to_B(A, B)
    img1 = im.immerge(np.concatenate([Ab1, Ab2, Ab3], axis=1))
    img2 = im.immerge(np.concatenate[b1A, b2A, b3A, b1b2, b2b3], axis=1)
    im.imwrite(img1, py.join(save_dir, 'sample_to_A', 'iter-%09d.png' % test_iter))
    im.imwrite(img2, py.join(save_dir, 'sample_to_B', 'iter-%09d.png' % test_iter))
