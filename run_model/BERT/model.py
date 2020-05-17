import numpy as np
import tensorflow as tf
import math
import gc
from bert import params_from_pretrained_ckpt, BertModelLayer, load_bert_weights
from tensorflow.keras import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Lambda, Dense, Input, concatenate, Flatten
from tensorflow.keras.backend import clear_session


__author__ = 'Nuttapong Mekvipad (n.mekvipad@hotmail.com)'


class DataGenerator(Sequence):

    def __init__(self, x, y, slice_index, batch_size):
        self.x, self.y = x, y
        self.slice_index = slice_index
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        x_idx = self.slice_index[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = [np.array(x), np.array(x_idx)]

        return batch_x, np.array(batch_y)


def make_slice_average_fn(ndim):
    def slice_by_tensor(x):
        def map_slice(y):
            to_slice = y[0]
            start = y[1]
            end = y[2]
            reduce_vec = tf.reduce_mean(tf.slice(to_slice, start, end), axis=0)
            out = tf.where(tf.math.is_nan(reduce_vec), 0.1, reduce_vec)
            return out

        matrix_to_slice = x[0]  # (batch_size, seq_len, emb_dim)
        index_tensor = x[1]  # (batch_size, idx-start-end)
        start_index = tf.slice(index_tensor, [0, 0],
                               [tf.shape(index_tensor)[0], 1])  # (batch_size, idx-start) tf.slice preserve dim
        end_index = tf.slice(index_tensor, [0, 1], [tf.shape(index_tensor)[0], 1])  # (batch_size, idx-end)

        index_start = tf.squeeze(tf.stack([start_index,
                                           tf.zeros(shape=tf.shape(start_index), dtype='int32')],
                                          axis=-1)
                                 )
        index_end = tf.squeeze(tf.stack([end_index - start_index,
                                         tf.fill(dims=tf.shape(start_index), value=ndim)],
                                        axis=-1)
                               )
        out_tensor = tf.map_fn(map_slice, [matrix_to_slice, index_start, index_end], dtype=tf.float32)
        out_tensor.set_shape([None, ndim])

        return out_tensor

    return slice_by_tensor


def make_gather_entity_border_fn(ndim):
    def gather_entity_border(x):
        # gather token at the border of entity and sentence
        # index_tensor is 2D tensor of entity border index

        matrix_to_slice = x[0]  # (batch_size, seq_len, emb_dim)
        index_tensor = x[1]  # (batch_size, idx-start-end)
        s = tf.shape(index_tensor)

        start_index = tf.slice(index_tensor, [0, 0],
                               [tf.shape(index_tensor)[0], 1])  # (batch_size, idx-start) tf.slice preserve dim
        end_index = tf.slice(index_tensor, [0, 1], [tf.shape(index_tensor)[0], 1])  # (batch_size, idx-end)

        index_start_gather = tf.squeeze(tf.stack([tf.expand_dims(tf.range(s[0]), axis=-1), start_index], axis=-1))
        index_end_gather = tf.squeeze(tf.stack([tf.expand_dims(tf.range(s[0]), axis=-1), end_index], axis=-1))

        index_gather = tf.stack([index_start_gather, index_end_gather], axis=1)
        out_tensor = tf.gather_nd(matrix_to_slice, index_gather)
        out_tensor.set_shape([None, 2, ndim])

        return out_tensor

    return gather_entity_border


def make_gather_entity_start_fn(ndim):
    def gather_entity_start(x):
        # gather token at the border of entity and sentence
        # index_tensor is 2D tensor of entity border index

        matrix_to_slice = x[0]  # (batch_size, seq_len, emb_dim)
        index_tensor = x[1]  # (batch_size, idx-start-end)
        s = tf.shape(index_tensor)

        start_index = tf.slice(index_tensor, [0, 0],
                               [tf.shape(index_tensor)[0], 1])  # (batch_size, idx-start) tf.slice preserve dim

        index_start_gather = tf.squeeze(tf.stack([tf.expand_dims(tf.range(s[0]), axis=-1), start_index], axis=-1))
        out_tensor = tf.gather_nd(matrix_to_slice, index_start_gather)
        out_tensor.set_shape([None, ndim])

        return out_tensor

    return gather_entity_start


def make_entity_average_model(bert_path, ckpt_file, max_seq_len, bert_dim):
    model_ckpt = bert_path + ckpt_file
    bert_params = params_from_pretrained_ckpt(bert_path)
    bert_layer = BertModelLayer.from_params(bert_params, name="bert", trainable=False)
    slice_fn = make_slice_average_fn(bert_dim)

    input_ids = Input(shape=(max_seq_len,), dtype='int32')
    index_ent1 = Input(shape=(2,), dtype='int32')
    index_ent2 = Input(shape=(2,), dtype='int32')
    bert_emb = bert_layer(input_ids)
    ent1_avg_emb = Lambda(lambda x: slice_fn(x))([bert_emb, index_ent1])
    ent2_avg_emb = Lambda(lambda x: slice_fn(x))([bert_emb, index_ent2])
    concat = concatenate([ent1_avg_emb, ent2_avg_emb])
    output = Dense(2, activation='softmax')(concat)
    model = Model(inputs=[input_ids, index_ent1, index_ent2], outputs=output)
    model.build(input_shape=(None, max_seq_len))

    load_bert_weights(bert_layer, model_ckpt)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def make_entity_start_model(bert_path, ckpt_file, max_seq_len, bert_dim):
    model_ckpt = bert_path + ckpt_file
    bert_params = params_from_pretrained_ckpt(bert_path)
    bert_layer = BertModelLayer.from_params(bert_params, name="bert", trainable=True)
    slice_fn = make_gather_entity_start_fn(bert_dim)

    input_ids = Input(shape=(max_seq_len,), dtype='int32')
    index_ent1 = Input(shape=(2,), dtype='int32')
    index_ent2 = Input(shape=(2,), dtype='int32')
    bert_emb = bert_layer(input_ids)
    ent1_start = Lambda(lambda x: slice_fn(x))([bert_emb, index_ent1])
    ent2_start = Lambda(lambda x: slice_fn(x))([bert_emb, index_ent2])
    concat = concatenate([ent1_start, ent2_start])
    output = Dense(2, activation='softmax')(concat)
    model = Model(inputs=[input_ids, index_ent1, index_ent2], outputs=output)
    model.build(input_shape=(None, max_seq_len))

    load_bert_weights(bert_layer, model_ckpt)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def make_all_section_average_model(bert_path, ckpt_file, max_seq_len, bert_dim):
    model_ckpt = bert_path + ckpt_file
    bert_params = params_from_pretrained_ckpt(bert_path)
    bert_layer = BertModelLayer.from_params(bert_params, name="bert", trainable=False)
    slice_fn = make_slice_average_fn(bert_dim)

    input_ids = Input(shape=(max_seq_len,), dtype='int32')
    index_pre_ent1 = Input(shape=(2,), dtype='int32')
    index_ent1 = Input(shape=(2,), dtype='int32')
    index_ent1_ent2 = Input(shape=(2,), dtype='int32')
    index_ent2 = Input(shape=(2,), dtype='int32')
    index_post_ent2 = Input(shape=(2,), dtype='int32')

    bert_emb = bert_layer(input_ids)
    pre_ent1_avg_emb = Lambda(lambda x: slice_fn(x))([bert_emb, index_pre_ent1])
    ent1_avg_emb = Lambda(lambda x: slice_fn(x))([bert_emb, index_ent1])
    ent1_ent2_avg_emb = Lambda(lambda x: slice_fn(x))([bert_emb, index_ent1_ent2])
    ent2_avg_emb = Lambda(lambda x: slice_fn(x))([bert_emb, index_ent2])
    post_ent2_avg_emb = Lambda(lambda x: slice_fn(x))([bert_emb, index_post_ent2])

    concat = concatenate([pre_ent1_avg_emb, ent1_avg_emb, ent1_ent2_avg_emb, ent2_avg_emb, post_ent2_avg_emb])
    output = Dense(2, activation='softmax')(concat)
    model = Model(inputs=[input_ids, index_pre_ent1, index_ent1, index_ent1_ent2, index_ent2, index_post_ent2],
                  outputs=output)
    model.build(input_shape=(None, max_seq_len))

    load_bert_weights(bert_layer, model_ckpt)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def make_cls_model(bert_path, ckpt_file, max_seq_len, bert_dim):
    model_ckpt = bert_path + ckpt_file
    bert_params = params_from_pretrained_ckpt(bert_path)
    bert_layer = BertModelLayer.from_params(bert_params, name="bert", trainable=False)

    input_ids = Input(shape=(max_seq_len,), dtype='int32')
    bert_emb = bert_layer(input_ids)
    cls = Lambda(lambda x: tf.gather(x, indices=0, axis=1))(bert_emb)
    output = Dense(2, activation='softmax')(cls)
    model = Model(inputs=input_ids, outputs=output)
    model.build(input_shape=(None, max_seq_len))

    load_bert_weights(bert_layer, model_ckpt)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def make_entity_border_model(bert_path, ckpt_file, max_seq_len, bert_dim):
    model_ckpt = bert_path + ckpt_file
    bert_params = params_from_pretrained_ckpt(bert_path)
    bert_layer = BertModelLayer.from_params(bert_params, name="bert", trainable=False)
    gather_fn = make_gather_entity_border_fn(bert_dim)

    input_ids = Input(shape=(max_seq_len,), dtype='int32')
    index_border_ent1 = Input(shape=(2,), dtype='int32')
    index_border_ent2 = Input(shape=(2,), dtype='int32')
    bert_emb = bert_layer(input_ids)
    ent1_avg_emb = Lambda(lambda x: gather_fn(x))([bert_emb, index_border_ent1])
    ent2_avg_emb = Lambda(lambda x: gather_fn(x))([bert_emb, index_border_ent2])
    ent1_flatten = Flatten()(ent1_avg_emb)
    ent2_flatten = Flatten()(ent2_avg_emb)
    concat = concatenate([ent1_flatten, ent2_flatten])
    output = Dense(2, activation='softmax')(concat)
    model = Model(inputs=[input_ids, index_border_ent1, index_border_ent2], outputs=output)
    model.build(input_shape=(None, max_seq_len))

    load_bert_weights(bert_layer, model_ckpt)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def make_entity_average_encoder(bert_path, ckpt_file, max_seq_len, bert_dim):
    model_ckpt = bert_path + ckpt_file
    bert_params = params_from_pretrained_ckpt(bert_path)
    bert_layer = BertModelLayer.from_params(bert_params, name="bert", trainable=False)
    slice_fn = make_slice_average_fn(bert_dim)

    input_ids = Input(shape=(max_seq_len,), dtype='int32')
    index_ent1 = Input(shape=(2,), dtype='int32')
    index_ent2 = Input(shape=(2,), dtype='int32')
    bert_emb = bert_layer(input_ids)
    ent1_avg_emb = Lambda(lambda x: slice_fn(x))([bert_emb, index_ent1])
    ent2_avg_emb = Lambda(lambda x: slice_fn(x))([bert_emb, index_ent2])
    output = concatenate([ent1_avg_emb, ent2_avg_emb])
    model = Model(inputs=[input_ids, index_ent1, index_ent2], outputs=output)
    model.build(input_shape=(None, max_seq_len))

    load_bert_weights(bert_layer, model_ckpt)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def make_entity_start_encoder(bert_path, ckpt_file, max_seq_len, bert_dim):
    model_ckpt = bert_path + ckpt_file
    bert_params = params_from_pretrained_ckpt(bert_path)
    bert_layer = BertModelLayer.from_params(bert_params, name="bert", trainable=False)
    slice_fn = make_gather_entity_start_fn(bert_dim)

    input_ids = Input(shape=(max_seq_len,), dtype='int32')
    index_ent1 = Input(shape=(2,), dtype='int32')
    index_ent2 = Input(shape=(2,), dtype='int32')
    bert_emb = bert_layer(input_ids)
    ent1_start = Lambda(lambda x: slice_fn(x))([bert_emb, index_ent1])
    ent2_start = Lambda(lambda x: slice_fn(x))([bert_emb, index_ent2])
    output = concatenate([ent1_start, ent2_start])
    model = Model(inputs=[input_ids, index_ent1, index_ent2], outputs=output)
    model.build(input_shape=(None, max_seq_len))

    load_bert_weights(bert_layer, model_ckpt)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def make_all_section_average_encoder(bert_path, ckpt_file, max_seq_len, bert_dim):
    model_ckpt = bert_path + ckpt_file
    bert_params = params_from_pretrained_ckpt(bert_path)
    bert_layer = BertModelLayer.from_params(bert_params, name="bert", trainable=False)
    slice_fn = make_slice_average_fn(bert_dim)

    input_ids = Input(shape=(max_seq_len,), dtype='int32')
    index_pre_ent1 = Input(shape=(2,), dtype='int32')
    index_ent1 = Input(shape=(2,), dtype='int32')
    index_ent1_ent2 = Input(shape=(2,), dtype='int32')
    index_ent2 = Input(shape=(2,), dtype='int32')
    index_post_ent2 = Input(shape=(2,), dtype='int32')

    bert_emb = bert_layer(input_ids)
    pre_ent1_avg_emb = Lambda(lambda x: slice_fn(x))([bert_emb, index_pre_ent1])
    ent1_avg_emb = Lambda(lambda x: slice_fn(x))([bert_emb, index_ent1])
    ent1_ent2_avg_emb = Lambda(lambda x: slice_fn(x))([bert_emb, index_ent1_ent2])
    ent2_avg_emb = Lambda(lambda x: slice_fn(x))([bert_emb, index_ent2])
    post_ent2_avg_emb = Lambda(lambda x: slice_fn(x))([bert_emb, index_post_ent2])

    output = concatenate([pre_ent1_avg_emb, ent1_avg_emb, ent1_ent2_avg_emb, ent2_avg_emb, post_ent2_avg_emb])
    model = Model(inputs=[input_ids, index_pre_ent1, index_ent1, index_ent1_ent2, index_ent2, index_post_ent2],
                  outputs=output)
    model.build(input_shape=(None, max_seq_len))

    load_bert_weights(bert_layer, model_ckpt)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def make_cls_encoder(bert_path, ckpt_file, max_seq_len, bert_dim):
    model_ckpt = bert_path + ckpt_file
    bert_params = params_from_pretrained_ckpt(bert_path)
    bert_layer = BertModelLayer.from_params(bert_params, name="bert", trainable=False)

    input_ids = Input(shape=(max_seq_len,), dtype='int32')
    bert_emb = bert_layer(input_ids)
    output = Lambda(lambda x: tf.gather(x, indices=0, axis=1))(bert_emb)

    model = Model(inputs=input_ids, outputs=output)
    model.build(input_shape=(None, max_seq_len))

    load_bert_weights(bert_layer, model_ckpt)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def make_entity_border_encoder(bert_path, ckpt_file, max_seq_len, bert_dim):
    model_ckpt = bert_path + ckpt_file
    bert_params = params_from_pretrained_ckpt(bert_path)
    bert_layer = BertModelLayer.from_params(bert_params, name="bert", trainable=False)
    gather_fn = make_gather_entity_border_fn(bert_dim)

    input_ids = Input(shape=(max_seq_len,), dtype='int32')
    index_border_ent1 = Input(shape=(2,), dtype='int32')
    index_border_ent2 = Input(shape=(2,), dtype='int32')
    bert_emb = bert_layer(input_ids)
    ent1_avg_emb = Lambda(lambda x: gather_fn(x))([bert_emb, index_border_ent1])
    ent2_avg_emb = Lambda(lambda x: gather_fn(x))([bert_emb, index_border_ent2])
    ent1_flatten = Flatten()(ent1_avg_emb)
    ent2_flatten = Flatten()(ent2_avg_emb)
    output = concatenate([ent1_flatten, ent2_flatten])

    model = Model(inputs=[input_ids, index_border_ent1, index_border_ent2], outputs=output)
    model.build(input_shape=(None, max_seq_len))

    load_bert_weights(bert_layer, model_ckpt)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def reset_keras(model):
    clear_session()
    del model
    print(gc.collect())

