import tensorflow as tf
import gc
import tempfile
import gzip
import time
import os
from run_model.BERT.preprocess import make_non_entities_interval, make_entity_border
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
from tensorflow.keras.backend import clear_session
from cocoscore.tagger.co_occurrence_score import _compute_metric, co_occurrence_score


def cv_keras_cocoscore(data_df, x, y, n_fold, model_fn, model_params_set):
    kf = KFold(n_splits=n_fold)
    tokens, ent1_position, ent2_position, all_len = x
    df = data_df[['pmid', 'paragraph', 'sentence', 'entity1', 'entity2', 'class']]
    performance_score = list()

    for params in model_params_set:
        params_score = list()
        for train_index, val_index in kf.split(data_df):
            train_df, train_tokens, train_ent1_pos, train_ent2_pos = df.iloc[train_index], tokens[train_index], \
                                                                     ent1_position[train_index], \
                                                                     ent2_position[train_index]

            val_df, val_tokens, val_ent1_pos, val_ent2_pos = df.iloc[val_index], tokens[val_index], \
                                                             ent1_position[val_index], ent2_position[val_index]

            train_y = to_categorical(y[train_index])

            if params['input_type'] == 'tokens':
                train_x = [train_tokens]
                val_x = [val_tokens]
            elif params['input_type'] == 'tokens-start-end':
                train_x = [train_tokens, train_ent1_pos, train_ent2_pos]
                val_x = [val_tokens, val_ent1_pos, val_ent2_pos]
            elif params['input_type'] == 'tokens-section':
                train_all_len = all_len[train_index]
                pre_ent1, ent1_ent2, post_ent2 = make_non_entities_interval(train_ent1_pos, train_ent2_pos,
                                                                            train_all_len)
                train_x = [train_tokens, pre_ent1, train_ent1_pos, ent1_ent2, train_ent2_pos, post_ent2]

                val_all_len = all_len[val_index]
                val_pre_ent1, val_ent1_ent2, val_post_ent2 = make_non_entities_interval(val_ent1_pos, val_ent2_pos,
                                                                                        val_all_len)

                val_x = [val_tokens, val_pre_ent1, val_ent1_pos, val_ent1_ent2, val_ent2_pos, val_post_ent2]

            elif params['input_type'] == 'tokens-border':
                ent1_border, ent2_border = make_entity_border(train_ent1_pos, train_ent2_pos)
                train_x = [train_tokens, ent1_border, ent2_border]

                val_ent1_border, val_ent2_border = make_entity_border(val_ent1_pos, val_ent2_pos)
                val_x = [val_tokens, val_ent1_border, val_ent2_border]
            else:
                raise ValueError('Wrong input_type parameter was specified.')

            model = model_fn(bert_path=params['bert_path'], ckpt_file=params['ckpt_file'],
                             max_seq_len=params['max_seq_len'], bert_dim=params['bert_dim'])
            model.fit(train_x, train_y, epochs=params['epochs'], batch_size=params['batch_size'])

            val_probabilities = model.predict(val_x)
            val_df = val_df.assign(predicted=val_probabilities[:, 1])

            _, tmp_file_path = tempfile.mkstemp(text=True, suffix='.gz')

            with gzip.open(tmp_file_path, 'wt') as test_out:
                val_df.to_csv(test_out, sep='\t', header=False, index=False,
                              columns=['pmid', 'paragraph', 'sentence', 'entity1', 'entity2', 'predicted'])

            val_score_dict = co_occurrence_score(matches_file_path=None,
                                                 score_file_path=tmp_file_path,
                                                 entities_file=None,
                                                 first_type=0,
                                                 second_type=0,
                                                 ignore_scores=False,
                                                 silent=True,
                                                 **params['cocoscore_params'],
                                                 )

            val_performance = _compute_metric(val_score_dict, val_df, warn=params['warn_missing_scores'],
                                              metric=params['metric'])

            params_score.append(val_performance)
            print(val_performance)
            clear_session()
            del model
            print(gc.collect())

        performance_score.append(params_score)

    return performance_score


def train_model(x, y, model_params, model_fn, input_type, checkpoint_file=None,
                create_checkpoint=True, checkpoint_path='./checkpoint_file/training-{epoch:04d}.ckpt', frequency=2):
    tokens, ent1_position, ent2_position, all_len = x
    train_y = to_categorical(y)

    if input_type == 'tokens':
        train_x = [tokens]

    elif input_type == 'tokens-start-end':
        train_x = [tokens, ent1_position, ent2_position]

    elif input_type == 'tokens-section':
        pre_ent1, ent1_ent2, post_ent2 = make_non_entities_interval(ent1_position, ent2_position,
                                                                    all_len)
        train_x = [tokens, pre_ent1, ent1_position, ent1_ent2, ent2_position, post_ent2]

    elif input_type == 'tokens-border':
        ent1_border, ent2_border = make_entity_border(ent1_position, ent2_position)
        train_x = [tokens, ent1_border, ent2_border]
    else:
        raise ValueError('Wrong input_type parameter was specified.')

    if checkpoint_file is not None:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1,
                                                         period=frequency)
        callbacks = [cp_callback]

        model = model_fn(bert_path=model_params['bert_path'], ckpt_file=model_params['ckpt_file'],
                         max_seq_len=model_params['max_seq_len'], bert_dim=model_params['bert_dim'])
        model.load_weights(checkpoint_file)
        model.fit(train_x, train_y, epochs=model_params['epochs'], batch_size=model_params['batch_size'],
                  callbacks=[cp_callback])

    elif create_checkpoint:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1,
                                                         period=frequency)
        callbacks = [cp_callback]
        model = model_fn(bert_path=model_params['bert_path'], ckpt_file=model_params['ckpt_file'],
                         max_seq_len=model_params['max_seq_len'], bert_dim=model_params['bert_dim'])
        model.fit(train_x, train_y, epochs=model_params['epochs'], batch_size=model_params['batch_size'],
                  callbacks=[cp_callback])

    else:
        model = model_fn(bert_path=model_params['bert_path'], ckpt_file=model_params['ckpt_file'],
                         max_seq_len=model_params['max_seq_len'], bert_dim=model_params['bert_dim'])
        model.fit(train_x, train_y, epochs=model_params['epochs'], batch_size=model_params['batch_size'])

    return model


def eval_model(data_df, x, model, cocoscore_params, input_type,
               warn_missing_scores=False, metric='roc_auc_score', return_cocoscores=False, baseline=False,
               return_dataframe=False):
    tokens, ent1_position, ent2_position, all_len = x
    df = data_df[['pmid', 'paragraph', 'sentence', 'entity1', 'entity2', 'class']]

    if baseline:
        df = df.assign(predicted=1)
    else:
        if input_type == 'tokens':
            val_x = [tokens]

        elif input_type == 'tokens-start-end':
            val_x = [tokens, ent1_position, ent2_position]

        elif input_type == 'tokens-section':
            pre_ent1, ent1_ent2, post_ent2 = make_non_entities_interval(ent1_position, ent2_position,
                                                                        all_len)
            val_x = [tokens, pre_ent1, ent1_position, ent1_ent2, ent2_position, post_ent2]

        elif input_type == 'tokens-border':
            ent1_border, ent2_border = make_entity_border(ent1_position, ent2_position)
            val_x = [tokens, ent1_border, ent2_border]
        else:
            raise ValueError('Wrong input_type parameter was specified.')

        val_probabilities = model.predict(val_x)
        df = df.assign(predicted=val_probabilities[:, 1])

    _, tmp_file_path = tempfile.mkstemp(text=True, suffix='.gz')

    with gzip.open(tmp_file_path, 'wt') as test_out:
        df.to_csv(test_out, sep='\t', header=False, index=False,
                  columns=['pmid', 'paragraph', 'sentence', 'entity1', 'entity2', 'predicted'])

    val_score_dict = co_occurrence_score(matches_file_path=None,
                                         score_file_path=tmp_file_path,
                                         entities_file=None,
                                         first_type=0,
                                         second_type=0,
                                         ignore_scores=False,
                                         silent=True,
                                         **cocoscore_params,
                                         )

    val_performance = _compute_metric(val_score_dict, df, warn=warn_missing_scores,
                                      metric=metric)

    if return_cocoscores and return_dataframe:
        return val_performance, val_score_dict, df
    elif return_cocoscores:
        return val_performance, val_score_dict
    elif return_dataframe:
        return val_performance, df
    else:
        return val_performance

def cv_logit_cocoscore(data_df, x, y, n_fold, encoder_params, logistic_params_set, cocoscore_params,
                       model_fn=None, baseline=False):
    kf = KFold(n_splits=n_fold)
    tokens, ent1_position, ent2_position, all_len = x
    df = data_df[['pmid', 'paragraph', 'sentence', 'entity1', 'entity2', 'class']]
    performance_score = list()

    start_time = time.time()
    if not baseline:
        if encoder_params['input_type'] == 'tokens':
            data_x = [tokens]
        elif encoder_params['input_type'] == 'tokens-start-end':
            data_x = [tokens, ent1_position, ent2_position]
        elif encoder_params['input_type'] == 'tokens-section':
            pre_ent1, ent1_ent2, post_ent2 = make_non_entities_interval(ent1_position, ent2_position, all_len)
            data_x = [tokens, pre_ent1, ent1_position, ent1_ent2, ent2_position, post_ent2]
        elif encoder_params['input_type'] == 'tokens-border':
            ent1_border, ent2_border = make_entity_border(ent1_position, ent2_position)
            data_x = [tokens, ent1_border, ent2_border]
        else:
            raise ValueError('Wrong input_type parameter was specified.')

        model = model_fn(bert_path=encoder_params['bert_path'], ckpt_file=encoder_params['ckpt_file'],
                         max_seq_len=encoder_params['max_seq_len'], bert_dim=encoder_params['bert_dim'])
        encoded_data = model.predict(data_x)
        clear_session()
        del model
        print(gc.collect())
    encoding_time = time.time()
    print('Finish encoding at ', encoding_time - start_time)

    for i, params in enumerate(logistic_params_set):
        params_score = list()
        for j, (train_index, val_index) in enumerate(kf.split(data_df)):
            if not baseline:
                train_df, encoded_train = df.iloc[train_index], encoded_data[train_index]
                val_df, encoded_test = df.iloc[val_index], encoded_data[val_index]
                train_y = y[train_index]
                clf = LogisticRegression(**params)
                clf.fit(encoded_train, train_y)
                val_probabilities = clf.predict_proba(encoded_test)
                val_df = val_df.assign(predicted=val_probabilities[:, 1])
            else:
                val_df = df.iloc[val_index]
                val_df = val_df.assign(predicted=1)

            _, tmp_file_path = tempfile.mkstemp(text=True, suffix='.gz')

            with gzip.open(tmp_file_path, 'wt') as test_out:
                val_df.to_csv(test_out, sep='\t', header=False, index=False,
                              columns=['pmid', 'paragraph', 'sentence', 'entity1', 'entity2', 'predicted'])

            val_score_dict = co_occurrence_score(matches_file_path=None,
                                                 score_file_path=tmp_file_path,
                                                 entities_file=None,
                                                 first_type=0,
                                                 second_type=0,
                                                 ignore_scores=False,
                                                 silent=True,
                                                 **cocoscore_params['cocoscore_params'],
                                                 )

            val_performance = _compute_metric(val_score_dict, val_df, warn=cocoscore_params['warn_missing_scores'],
                                              metric=cocoscore_params['metric'])
            end_time = time.time()
            print('Finish {}th round of CV for {}th parameter set with time = '.format(j, i), end_time - start_time)
            params_score.append(val_performance)

        performance_score.append(params_score)

    return performance_score


def train_logit_bert(x, y, encoder_params, logistic_params, encoder_fn=None):
    tokens, ent1_position, ent2_position, all_len = x
    if encoder_params['input_type'] == 'tokens':
        data_x = [tokens]
    elif encoder_params['input_type'] == 'tokens-start-end':
        data_x = [tokens, ent1_position, ent2_position]
    elif encoder_params['input_type'] == 'tokens-section':
        pre_ent1, ent1_ent2, post_ent2 = make_non_entities_interval(ent1_position, ent2_position, all_len)
        data_x = [tokens, pre_ent1, ent1_position, ent1_ent2, ent2_position, post_ent2]
    elif encoder_params['input_type'] == 'tokens-border':
        ent1_border, ent2_border = make_entity_border(ent1_position, ent2_position)
        data_x = [tokens, ent1_border, ent2_border]
    else:
        raise ValueError('Wrong input_type parameter was specified.')

    encoder = encoder_fn(bert_path=encoder_params['bert_path'], ckpt_file=encoder_params['ckpt_file'],
                         max_seq_len=encoder_params['max_seq_len'], bert_dim=encoder_params['bert_dim'])
    encoded_data = encoder.predict(data_x)
    clear_session()
    del encoder
    print(gc.collect())

    clf = LogisticRegression(**logistic_params)
    clf.fit(encoded_data, y)

    return clf


def eval_logit_bert(data_df, x, model, cocoscore_params, encoder_params=None, encoder_fn=None,
                    warn_missing_scores=False, metric='roc_auc_score',
                    return_cocoscores=False, baseline=False, return_dataframe=False):
    if baseline:
        data_df = data_df.assign(predicted=1)
    else:
        tokens, ent1_position, ent2_position, all_len = x
        if encoder_params['input_type'] == 'tokens':
            data_x = [tokens]
        elif encoder_params['input_type'] == 'tokens-start-end':
            data_x = [tokens, ent1_position, ent2_position]
        elif encoder_params['input_type'] == 'tokens-section':
            pre_ent1, ent1_ent2, post_ent2 = make_non_entities_interval(ent1_position, ent2_position, all_len)
            data_x = [tokens, pre_ent1, ent1_position, ent1_ent2, ent2_position, post_ent2]
        elif encoder_params['input_type'] == 'tokens-border':
            ent1_border, ent2_border = make_entity_border(ent1_position, ent2_position)
            data_x = [tokens, ent1_border, ent2_border]
        else:
            raise ValueError('Wrong input_type parameter was specified.')

        encoder = encoder_fn(bert_path=encoder_params['bert_path'], ckpt_file=encoder_params['ckpt_file'],
                             max_seq_len=encoder_params['max_seq_len'], bert_dim=encoder_params['bert_dim'])
        encoded_data = encoder.predict(data_x)
        clear_session()
        del encoder
        print(gc.collect())
        probabilities = model.predict_proba(encoded_data)
        data_df = data_df.assign(predicted=probabilities[:, 1])

    _, tmp_file_path = tempfile.mkstemp(text=True, suffix='.gz')

    with gzip.open(tmp_file_path, 'wt') as test_out:
        data_df.to_csv(test_out, sep='\t', header=False, index=False,
                       columns=['pmid', 'paragraph', 'sentence', 'entity1', 'entity2', 'predicted'])

    val_score_dict = co_occurrence_score(matches_file_path=None,
                                         score_file_path=tmp_file_path,
                                         entities_file=None,
                                         first_type=0,
                                         second_type=0,
                                         ignore_scores=False,
                                         silent=True,
                                         **cocoscore_params
                                         )

    val_performance = _compute_metric(val_score_dict, data_df, warn=warn_missing_scores,
                                      metric=metric)
    if return_cocoscores and return_dataframe:
        return val_performance, val_score_dict, data_df
    elif return_cocoscores:
        return val_performance, val_score_dict
    elif return_dataframe:
        return val_performance, data_df
    else:
        return val_performance
