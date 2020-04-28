import functools

import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tensorflow.keras as keras
import tf2lib as tl
import tf2gan as gan
import tqdm

import data
import module


# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--dataset', default='digit-data')
py.arg('--datasets_dir', default='digit-data')
py.arg('--load_size', type=int, default=36)  # load image to this size
py.arg('--crop_size', type=int, default=32)  # then crop to this size
py.arg('--batch_size', type=int, default=1)
py.arg('--img_size', type=int, default=32)
py.arg('--epochs', type=int, default=200)
py.arg('--epoch_decay', type=int, default=100)  # epoch to start decaying learning rate
py.arg('--lr', type=float, default=0.0002)
py.arg('--beta_1', type=float, default=0.5)
py.arg('--adversarial_loss_mode', default='lsgan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
py.arg('--gradient_penalty_mode', default='none', choices=['none', 'dragan', 'wgan-gp'])
py.arg('--gradient_penalty_weight', type=float, default=10.0)
py.arg('--cycle_loss_weight', type=float, default=10.0)
py.arg('--identity_loss_weight', type=float, default=0.0)
py.arg('--attr_loss_weight', type=float, default=5.0)
py.arg('--rest_loss_weight', type=float, default=5.0)
py.arg('--rec_loss_weight', type=float, default=10.0)
py.arg('--pool_size', type=int, default=50)  # pool size to store fake samples
args = py.args()

# output_dir
output_dir = py.join('output', args.dataset)
py.mkdir(output_dir)

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)


# ==============================================================================
# =                                    data                                    =
# ==============================================================================

real_img_paths = py.glob(py.join(args.datasets_dir, 'real-digits', 'train'), '*.png')
syn_img_paths = py.glob(py.join(args.datasets_dir, 'syn-digits', 'train'), '*.png') # default does not search subdirectories
# b1_img_paths = py.glob(py.join(args.datasets_dir, 'syn-digits', 'train', 'b1'), '*.png')
# b2_img_paths = py.glob(py.join(args.datasets_dir, 'syn-digits', 'train', 'b2'), '*.png')
# b3_img_paths = py.glob(py.join(args.datasets_dir, 'syn-digits', 'train', 'b3'), '*.png')
A_B_dataset, len_dataset = data.make_zip_dataset(real_img_paths, syn_img_paths, args.batch_size, args.img_size, args.img_size, training=True, repeat=False)

A2B_pool = data.ItemPool(args.pool_size)
B2A_pool = data.ItemPool(args.pool_size)

real_img_paths_test = py.glob(py.join(args.datasets_dir, 'real-digits', 'test'), '*.png')
syn_img_paths_test = py.glob(py.join(args.datasets_dir, 'syn-digits', 'test'), '*.png')
# b1_img_paths_test = py.glob(py.join(args.datasets_dir, 'syn-digits', 'test', 'b1'), '*.png')
# b2_img_paths_test = py.glob(py.join(args.datasets_dir, 'syn-digits', 'test', 'b2'), '*.png')
# b3_img_paths_test = py.glob(py.join(args.datasets_dir, 'syn-digits', 'test', 'b3'), '*.png')
A_B_dataset_test, _ = data.make_zip_dataset(real_img_paths_test, syn_img_paths_test, args.batch_size, args.img_size, args.img_size, training=False, repeat=True)


# ==============================================================================
# =                                   models                                   =
# ==============================================================================

G_A2B = module.ResnetGenerator(input_shape=(args.img_size, args.img_size, 3))
G_B2A = module.ResnetGenerator(input_shape=(args.img_size, args.img_size, 3))

full_embed = module.compress(input_shape=(args.img_size, args.img_size, 3))

decode_A = module.ConvDiscriminator(input_shape=(args.img_size, args.img_size, 3))
decode_B = module.ConvDiscriminator(input_shape=(args.img_size, args.img_size, 3))

d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn(args.adversarial_loss_mode)
cycle_loss_fn = tf.losses.MeanAbsoluteError()
identity_loss_fn = tf.losses.MeanAbsoluteError()
supervised_loss_fn = tf.losses.MeanAbsoluteError()

G_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
D_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
G_optimizer = keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=args.beta_1)
D_optimizer = keras.optimizers.Adam(learning_rate=D_lr_scheduler, beta_1=args.beta_1)


# ==============================================================================
# =                                 train step                                 =
# ==============================================================================

@tf.function
def train_G(A, B):
    with tf.GradientTape() as t:
        b1, b2, b3 = data.split_B(B)

        A2B = G_A2B(A, training=True)
        B12A = G_B2A(b1, training=True)
        B22A = G_B2A(b2, training=True)
        B32A = G_B2A(b3, training=True)
        A2B2A = G_B2A(A2B, training=True)
        B12A2B = G_A2B(B12A, training=True)
        B22A2B = G_A2B(B22A, training=True)
        B3A2B = G_A2B(B32A, training=True)
        A2A = G_B2A(A, training=True)
        B12B = G_A2B(b1, training=True)
        B22B = G_A2B(b2, training=True)
        B32B = G_A2B(b3, training=True)

        attr_emb_A = tf.slice(full_embed(A2A), [0,0], [1,64])
        rest_emb_A = tf.slice(full_embed(A2A), [0,64], [1,64])

        attr_emb_b1 = tf.slice(full_embed(B12A2B), [0,0], [1,64])
        rest_emb_b1 = tf.slice(full_embed(B12A2B), [0,64], [1.64])
        attr_emb_b2 = tf.slice(full_embed(B22A2B), [0,0], [1,64])
        rest_emb_b2 = tf.slice(full_embed(B22A2B), [0,64], [1,64])
        attr_emb_b3 = tf.slice(full_embed(B3A2B), [0,0], [1,64])
        rest_emb_b3 = tf.slice(full_embed(B3A2B), [0,64], [1,64])

        aa_D_A_logits = decode_A(tf.concat([attr_emb_A, rest_emb_A], 1), training=True)
        b1b1_D_A_logits = decode_A(tf.concat([attr_emb_b1, rest_emb_b1], 1), training=True)
        b1b2_D_A_logits = decode_A(tf.concat([attr_emb_b1, rest_emb_b2], 1), training=True)
        b1b3_D_A_logits = decode_A(tf.concat([attr_emb_b1, rest_emb_b3], 1), training=True)
        b2b1_D_A_logits = decode_A(tf.concat([attr_emb_b2, rest_emb_b1], 1), training=True)
        b2b2_D_A_logits = decode_A(tf.concat([attr_emb_b2, rest_emb_b2], 1), training=True)
        b2b3_D_A_logits = decode_A(tf.concat([attr_emb_b2, rest_emb_b3], 1), training=True)
        b3b1_D_A_logits = decode_A(tf.concat([attr_emb_b3, rest_emb_b3], 1), training=True)
        b3b2_D_A_logits = decode_A(tf.concat([attr_emb_b3, rest_emb_b3], 1), training=True)
        b3b3_D_A_logits = decode_A(tf.concat([attr_emb_b3, rest_emb_b3], 1), training=True)

        aa_D_B_logits = decode_B(tf.concat([attr_emb_A, rest_emb_A], 1), training=True)
        b1b1_D_B_logits = decode_B(tf.concat([attr_emb_b1, rest_emb_b1], 1), training=True)
        b1b2_D_B_logits = decode_B(tf.concat([attr_emb_b1, rest_emb_b2], 1), training=True)
        b1b3_D_B_logits = decode_B(tf.concat([attr_emb_b1, rest_emb_b3], 1), training=True)
        b2b1_D_B_logits = decode_B(tf.concat([attr_emb_b2, rest_emb_b1], 1), training=True)
        b2b2_D_B_logits = decode_B(tf.concat([attr_emb_b2, rest_emb_b2], 1), training=True)
        b2b3_D_B_logits = decode_B(tf.concat([attr_emb_b2, rest_emb_b3], 1), training=True)
        b3b1_D_B_logits = decode_B(tf.concat([attr_emb_b3, rest_emb_b1], 1), training=True)
        b3b2_D_B_logits = decode_B(tf.concat([attr_emb_b3, rest_emb_b2], 1), training=True)
        b3b3_D_B_logits = decode_B(tf.concat([attr_emb_b3, rest_emb_b3], 1), training=True)

        # GAN loss for C_A
        aa_A_g_loss = g_loss_fn(aa_D_A_logits)
        b1b1_A_g_loss = g_loss_fn(b1b1_D_A_logits)
        b1b2_A_g_loss = g_loss_fn(b1b2_D_A_logits)
        b1b3_A_g_loss = g_loss_fn(b1b3_D_A_logits)
        b2b1_A_g_loss = g_loss_fn(b2b1_D_A_logits)
        b2b2_A_g_loss = g_loss_fn(b2b2_D_A_logits)
        b2b3_A_g_loss = g_loss_fn(b2b3_D_A_logits)
        b3b1_A_g_loss = g_loss_fn(b3b1_D_A_logits)
        b3b2_A_g_loss = g_loss_fn(b3b2_D_A_logits)
        b3b3_A_g_loss = g_loss_fn(b3b3_D_A_logits)

        # GAN loss for C_B
        aa_B_g_loss = g_loss_fn(aa_D_B_logits)
        b1b1_B_g_loss = g_loss_fn(b1b1_D_B_logits)
        b1b2_B_g_loss = g_loss_fn(b1b2_D_B_logits)
        b1b3_B_g_loss = g_loss_fn(b1b3_D_B_logits)
        b2b1_B_g_loss = g_loss_fn(b2b1_D_B_logits)
        b2b2_B_g_loss = g_loss_fn(b2b2_D_B_logits)
        b2b3_B_g_loss = g_loss_fn(b2b3_D_B_logits)
        b3b1_B_g_loss = g_loss_fn(b3b1_D_B_logits)
        b3b2_B_g_loss = g_loss_fn(b3b2_D_B_logits)
        b3b3_B_g_loss = g_loss_fn(b3b3_D_B_logits)

        # # A2B and B2A cycle losses
        A2B2A_cycle_loss = cycle_loss_fn(A, A2B2A)
        b12A2b1_cycle_loss = cycle_loss_fn(b1,B12A2B)
        b22A2b2_cycle_loss = cycle_loss_fn(b2,B22A2B)
        b32A2b3_cycle_loss = cycle_loss_fn(b3,B32A2B)

        # compositional/attribue cycle losses
        a_tilde = tf.concat([attr_emb_b1, rest_emb_A], axis=1)
        b_tilde = tf.concat([attr_emb_A, rest_emb_b1], axis=1) # any b works for this because we're trying to constrain a to keep attr_emb_A
        a_comp_cycle_loss = cycle_loss_fn(A, \
                                        data.uncompress(tf.concat([tf.slice(full_embed(b_tilde), [0,64], [1,64]), rest_emb_A], axis=0)))
        b3_comp_cycle_loss = cycle_loss_fn(b3, \
                                        data.uncompress(tf.concat([tf.slice(full_embed(a_tilde), [0,0], [1,64]), rest_emb_b2], axis=0)))
        # a_comp_cycle_loss = cycle_loss_fn(tf.concat([attr_emb_A, rest_emb_A], axis=1), \
        #                                 tf.concat([full_embed(b_tilde)[:64], rest_emb_A], axis=1))
        # b3_comp_cycle_loss = cycle_loss_fn(tf.concat([attr_emb_b3, rest_emb_b3], axis=1), \
        #                                 tf.concat([full_embed(a_tilde)[:64], rest_emb_b2], axis=1))

        # identity losses
        A2A_id_loss = identity_loss_fn(A, A2A)
        b12b1_id_loss = identity_loss_fn(b1, B12B)
        b22b2_id_loss = identity_loss_fn(b2, B22B)
        b32b3_id_loss = identity_loss_fn(b3, B32B)

        # supervised loss on synth data
        b3_constr_loss = supervised_loss_fn(b3, tf.concat([attr_emb_b1, rest_emb_b2], axis=1))

        G_loss = (aa_A_g_loss + b1b1_A_g_loss + b1b2_A_g_loss + b1b3_A_g_loss + b2b1_A_g_loss + b2b2_A_g_loss + b2b3_A_g_loss + \
                b3b1_A_g_loss + b3b2_A_g_loss + b3b3_A_g_loss) + \
                (aa_B_g_loss + b1b1_B_g_loss + b1b2_B_g_loss + b1b3_B_g_loss + b2b1_B_g_loss + b2b2_B_g_loss + b2b3_B_g_loss + \
                b3b1_B_g_loss + b3b2_B_g_loss + b3b3_B_g_loss) + \
                (a_comp_cycle_loss + b3_comp_cycle_loss) * args.attr_loss_weight + \
                (A2A_id_loss + b12b1_id_loss + b22b2_id_loss + b32b3_id_loss) * args.identity_loss_weight
                # () * args.rest_loss_weight + \
                # (A2B2A_cycle_loss + b12A2b1_cycle_loss + b22A2b2_cycle_loss + b32A2b3_cycle_loss) * args.cycle_loss_weight + \

        # G_loss = (A2B_g_loss + B2A_g_loss) + (A2B2A_cycle_loss + B2A2B_cycle_loss) * args.cycle_loss_weight + (A2A_id_loss + B2B_id_loss) * args.identity_loss_weight

    G_grad = t.gradient(G_loss, attr_emb_A.trainable_variables + rest_emb_A.trainable_variables + \
                                attr_emb_b1.trainable_variables + rest_emb_b1.trainable_variables + \
                                attr_emb_b2.trainable_variables + rest_emb_b2.trainable_variables + \
                                attr_emb_b3.trainable_variables + rest_emb_b3.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, \
                                attr_emb_A.trainable_variables + rest_emb_A.trainable_variables + \
                                attr_emb_b1.trainable_variables + rest_emb_b1.trainable_variables + \
                                attr_emb_b2.trainable_variables + rest_emb_b2.trainable_variables + \
                                attr_emb_b3.trainable_variables + rest_emb_b3.trainable_variables))

    return attr_emb_A, rest_emb_A, \
            attr_emb_b1, rest_emb_b1, \
            attr_emb_b2, rest_emb_b2, \
            attr_emb_b3, rest_emb_b3, \
                {'A_g_loss': (aa_A_g_loss + b1b1_A_g_loss + b1b2_A_g_loss + b1b3_A_g_loss + b2b1_A_g_loss + b2b2_A_g_loss + b2b3_A_g_loss + \
                            b3b1_A_g_loss + b3b2_A_g_loss + b3b3_A_g_loss),
                'B_g_loss': (aa_B_g_loss + b1b1_B_g_loss + b1b2_B_g_loss + b1b3_B_g_loss + b2b1_B_g_loss + b2b2_B_g_loss + b2b3_B_g_loss + \
                            b3b1_B_g_loss + b3b2_B_g_loss + b3b3_B_g_loss),
                'A2B2A_cycle_loss': A2B2A_cycle_loss,
                'b12A2b1_cycle_loss': b12A2b1_cycle_loss,
                'b22A2b2_cycle_loss': b22A2b2_cycle_loss,
                'b32A2b3_cycle_loss': b32A2b3_cycle_loss,
                'a_comp_cycle_loss': a_comp_cycle_loss,
                'b3_comp_cycle_loss': b3_comp_cycle_loss,
                'A2A_id_loss': A2A_id_loss,
                'b12b1_id_loss': b12b1_id_loss,
                'b22b2_id_loss': b22b2_id_loss,
                'b32b3_id_loss': b32b3_id_loss}


@tf.function
def train_D(A, B):
    with tf.GradientTape() as t:
        b1, b2, b3 = data.split_B(B)

        attr_emb_A = tf.slice(full_embed(A2A), [0,0], [1,64])
        rest_emb_A = tf.slice(full_embed(A2A), [0,64], [1,64])

        attr_emb_b1 = tf.slice(full_embed(B12A2B), [0,0], [1,64])
        rest_emb_b1 = tf.slice(full_embed(B12A2B), [0,64], [1.64])
        attr_emb_b2 = tf.slice(full_embed(B22A2B), [0,0], [1,64])
        rest_emb_b2 = tf.slice(full_embed(B22A2B), [0,64], [1,64])
        attr_emb_b3 = tf.slice(full_embed(B3A2B), [0,0], [1,64])
        rest_emb_b3 = tf.slice(full_embed(B3A2B), [0,64], [1,64])
        
        # D_A logits
        aa_D_A_logits = decode_A(tf.concat([attr_emb_A, rest_emb_A], axis=1), training=True)
        b1b1_D_A_logits = decode_A(tf.concat([attr_emb_b1, rest_emb_b1], axis=1), training=True)
        b1b2_D_A_logits = decode_A(tf.concat([attr_emb_b1, rest_emb_b2], axis=1), training=True)
        b1b3_D_A_logits = decode_A(tf.concat([attr_emb_b1, rest_emb_b3], axis=1), training=True)
        b2b1_D_A_logits = decode_A(tf.concat([attr_emb_b2, rest_emb_b1], axis=1), training=True)
        b2b2_D_A_logits = decode_A(tf.concat([attr_emb_b2, rest_emb_b2], axis=1), training=True)
        b2b3_D_A_logits = decode_A(tf.concat([attr_emb_b2, rest_emb_b3], axis=1), training=True)
        b3b1_D_A_logits = decode_A(tf.concat([attr_emb_b3, rest_emb_b3], axis=1), training=True)
        b3b2_D_A_logits = decode_A(tf.concat([attr_emb_b3, rest_emb_b3], axis=1), training=True)
        b3b3_D_A_logits = decode_A(tf.concat([attr_emb_b3, rest_emb_b3], axis=1), training=True)

        # D_B logits
        aa_D_B_logits = decode_B(tf.concat([attr_emb_A, rest_emb_A], axis=1), training=True)
        b1b1_D_B_logits = decode_B(tf.concat([attr_emb_b1, rest_emb_b1], axis=1), training=True)
        b1b2_D_B_logits = decode_B(tf.concat([attr_emb_b1, rest_emb_b2], axis=1), training=True)
        b1b3_D_B_logits = decode_B(tf.concat([attr_emb_b1, rest_emb_b3], axis=1), training=True)
        b2b1_D_B_logits = decode_B(tf.concat([attr_emb_b2, rest_emb_b1], axis=1), training=True)
        b2b2_D_B_logits = decode_B(tf.concat([attr_emb_b2, rest_emb_b2], axis=1), training=True)
        b2b3_D_B_logits = decode_B(tf.concat([attr_emb_b2, rest_emb_b3], axis=1), training=True)
        b3b1_D_B_logits = decode_B(tf.concat([attr_emb_b3, rest_emb_b1], axis=1), training=True)
        b3b2_D_B_logits = decode_B(tf.concat([attr_emb_b3, rest_emb_b2], axis=1), training=True)
        b3b3_D_B_logits = decode_B(tf.concat([attr_emb_b3, rest_emb_b3], axis=1), training=True)

        # A_d with all other D_A losses (not sure if we need to permutate for all other possibilities)
        A_d_loss1, b1b1_d_loss1 = d_loss_fn(aa_D_A_logits, b1b1_D_A_logits)
        A_d_loss2, b1b2_d_loss1 = d_loss_fn(aa_D_A_logits, b1b2_D_A_logits)
        A_d_loss3, b1b3_d_loss1 = d_loss_fn(aa_D_A_logits, b1b3_D_A_logits)
        A_d_loss4, b2b1_d_loss1 = d_loss_fn(aa_D_A_logits, b2b1_D_A_logits)
        A_d_loss5, b2b2_d_loss1 = d_loss_fn(aa_D_A_logits, b2b2_D_A_logits)
        A_d_loss6, b2b3_d_loss1 = d_loss_fn(aa_D_A_logits, b2b3_D_A_logits)
        A_d_loss7, b3b1_d_loss1 = d_loss_fn(aa_D_A_logits, b3b1_D_A_logits)
        A_d_loss8, b3b2_d_loss1 = d_loss_fn(aa_D_A_logits, b3b2_D_A_logits)
        A_d_loss9, b3b3_d_loss1 = d_loss_fn(aa_D_A_logits, b3b3_D_A_logits)

        # B_d with all other D_A losses (not sure if we need to permutate for all other possibilities)
        B_d_loss1, b1b1_d_loss2 = d_loss_fn(aa_D_A_logits, b1b1_D_A_logits)
        B_d_loss2, b1b2_d_loss2 = d_loss_fn(aa_D_A_logits, b1b2_D_A_logits)
        B_d_loss3, b1b3_d_loss2 = d_loss_fn(aa_D_A_logits, b1b3_D_A_logits)
        B_d_loss4, b2b1_d_loss2 = d_loss_fn(aa_D_A_logits, b2b1_D_A_logits)
        B_d_loss5, b2b2_d_loss2 = d_loss_fn(aa_D_A_logits, b2b2_D_A_logits)
        B_d_loss6, b2b3_d_loss2 = d_loss_fn(aa_D_A_logits, b2b3_D_A_logits)
        B_d_loss7, b3b1_d_loss2 = d_loss_fn(aa_D_A_logits, b3b1_D_A_logits)
        B_d_loss8, b3b2_d_loss2 = d_loss_fn(aa_D_A_logits, b3b2_D_A_logits)
        B_d_loss9, b3b3_d_loss2 = d_loss_fn(aa_D_A_logits, b3b3_D_A_logits)

        # unsure how to translate this for our problem since we are not trying to translate between synth and real domains
        # D_A_gp = gan.gradient_penalty(functools.partial(decode_A, training=True), A, B2A, mode=args.gradient_penalty_mode)
        # D_B_gp = gan.gradient_penalty(functools.partial(decode_B, training=True), B, A2B, mode=args.gradient_penalty_mode)

        D_loss = (A_d_loss1 + b1b1_d_loss1 + A_d_loss2 + b1b2_d_loss1 + A_d_loss3 + b1b3_d_loss1 + A_d_loss4 + b2b1_d_loss1 + \
                A_d_loss5 + b2b2_d_loss1 + A_d_loss6 + b2b3_d_loss1 + A_d_loss7 + b3b1_d_loss1 + A_d_loss8 + b3b2_d_loss1 + A_d_loss9 + b3b3_d_loss1) + \
                (B_d_loss1 + b1b1_d_loss2 + B_d_loss2 + b1b2_d_loss2 + B_d_loss3 + b1b3_d_loss2 + B_d_loss4 + b2b1_d_loss2 + \
                B_d_loss5 + b2b2_d_loss2 + B_d_loss6 + b2b3_d_loss2 + B_d_loss7 + b3b1_d_loss2 + B_d_loss8 + b3b2_d_loss2 + B_d_loss9 + b3b3_d_loss2)
        # D_loss = (A_d_loss + B2A_d_loss) + (B_d_loss + A2B_d_loss) + (D_A_gp + D_B_gp) * args.gradient_penalty_weight

    D_grad = t.gradient(D_loss, decode_A.trainable_variables + decode_B.trainable_variables)
    D_optimizer.apply_gradients(zip(D_grad, decode_A.trainable_variables + decode_B.trainable_variables))

    return {'A_d_loss': (A_d_loss1 + b1b1_d_loss1 + A_d_loss2 + b1b2_d_loss1 + A_d_loss3 + b1b3_d_loss1 + A_d_loss4 + b2b1_d_loss1 + \
                A_d_loss5 + b2b2_d_loss1 + A_d_loss6 + b2b3_d_loss1 + A_d_loss7 + b3b1_d_loss1 + A_d_loss8 + b3b2_d_loss1 + A_d_loss9 + b3b3_d_loss1),
            'B_d_loss': (B_d_loss1 + b1b1_d_loss2 + B_d_loss2 + b1b2_d_loss2 + B_d_loss3 + b1b3_d_loss2 + B_d_loss4 + b2b1_d_loss2 + \
                B_d_loss5 + b2b2_d_loss2 + B_d_loss6 + b2b3_d_loss2 + B_d_loss7 + b3b1_d_loss2 + B_d_loss8 + b3b2_d_loss2 + B_d_loss9 + b3b3_d_loss2)}
            # 'D_A_gp': D_A_gp,
            # 'D_B_gp': D_B_gp}


def train_step(A, B):
    attr_emb_A, rest_emb_A, \
        attr_emb_b1, rest_emb_b1, \
        attr_emb_b2, rest_emb_b2, \
        attr_emb_b3, rest_emb_b3, \
        G_loss_dict = train_G(A, B)

    # # cannot autograph `A2B_pool`
    # A2B = A2B_pool(A2B)  # or A2B = A2B_pool(A2B.numpy()), but it is much slower
    # B2A = B2A_pool(B2A)  # because of the communication between CPU and GPU

    D_loss_dict = train_D(A, B)

    return G_loss_dict, D_loss_dict


# @tf.function
# def sample(A, B):
#     A2B = G_A2B(A, training=False)
#     B2A = G_B2A(B, training=False)
#     A2B2A = G_B2A(A2B, training=False)
#     B2A2B = G_A2B(B2A, training=False)
#     return A2B, B2A, A2B2A, B2A2B

@tf.function
def sample(A, B):
    b1, b2, b3 = data.split_B(B)

    attr_emb_A = tf.slice(full_embed(A2A), [0,0], [1,64])
    rest_emb_A = tf.slice(full_embed(A2A), [0,64], [1,64])

    attr_emb_b1 = tf.slice(full_embed(B12A2B), [0,0], [1,64])
    rest_emb_b1 = tf.slice(full_embed(B12A2B), [0,64], [1.64])
    attr_emb_b2 = tf.slice(full_embed(B22A2B), [0,0], [1,64])
    rest_emb_b2 = tf.slice(full_embed(B22A2B), [0,64], [1,64])
    attr_emb_b3 = tf.slice(full_embed(B3A2B), [0,0], [1,64])
    rest_emb_b3 = tf.slice(full_embed(B3A2B), [0,64], [1,64])

    A = tf.concat([attr_emb_A, rest_emb_A], axis=1)
    b3 = tf.concat([attr_emb_b1, rest_emb_b2],axis=1)
    Ab1 = tf.concat([attr_emb_b1, rest_emb_A], axis=1)
    Ab2 = tf.concat([attr_emb_b2, rest_emb_A], axis=1)
    Ab3 = tf.concat([attr_emb_b3, rest_emb_A], axis=1)

    return A, b3, Ab1, Ab2, Ab3


# ==============================================================================
# =                                    run                                     =
# ==============================================================================

# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
checkpoint = tl.Checkpoint(dict(full_embed=full_embed,
                                decode_A=decode_A,
                                decode_B=decode_B,
                                G_optimizer=G_optimizer,
                                D_optimizer=D_optimizer,
                                ep_cnt=ep_cnt),
                           py.join(output_dir, 'checkpoints'),
                           max_to_keep=5)
try:  # restore checkpoint including the epoch counter
    checkpoint.restore().assert_existing_objects_matched()
except Exception as e:
    print(e)

# summary
train_summary_writer = tf.summary.create_file_writer(py.join(output_dir, 'summaries', 'train'))

# sample
test_iter = iter(A_B_dataset_test)
sample_dir = py.join(output_dir, 'samples_training')
py.mkdir(sample_dir)

# main loop
with train_summary_writer.as_default():
    for ep in tqdm.trange(args.epochs, desc='Epoch Loop'):
        if ep < ep_cnt:
            continue

        # update epoch counter
        ep_cnt.assign_add(1)

        # train for an epoch
        for A, B in tqdm.tqdm(A_B_dataset, desc='Inner Epoch Loop', total=len_dataset):
            G_loss_dict, D_loss_dict = train_step(A, B)

            # # summary
            tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
            tl.summary(D_loss_dict, step=G_optimizer.iterations, name='D_losses')
            tl.summary({'learning rate': G_lr_scheduler.current_learning_rate}, step=G_optimizer.iterations, name='learning rate')

            # sample
            if G_optimizer.iterations.numpy() % 100 == 0:
                A, B = next(test_iter)
                # A2B, B2A, A2B2A, B2A2B = sample(A, B)
                # img = im.immerge(np.concatenate([A, A2B, A2B2A, B, B2A, B2A2B], axis=0), n_rows=2)
                img = im.immerge(np.concatenate([A, B], axis=0), n_rows=2)
                im.imwrite(img, py.join(sample_dir, 'iter-%09d.jpg' % G_optimizer.iterations.numpy()))

        # save checkpoint
        checkpoint.save(ep)
