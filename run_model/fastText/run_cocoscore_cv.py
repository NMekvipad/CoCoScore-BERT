import argparse
import pickle
import pandas as pd
import tempfile
import sys
import gzip
import cocoscore.ml.fasttext_helpers as fth
from cocoscore.ml.fasttext_helpers import fasttext_fit
from sklearn.model_selection import KFold
from cocoscore.tagger.co_occurrence_score import _compute_metric, co_occurrence_score

__author__ = 'Nuttapong Mekvipad (n.mekvipad@hotmail.com)'

sys.path.append('/home/projects/jensenlab/people/nutmek/fastText/fasttext')

parser = argparse.ArgumentParser(
    description="""
    Run multiple CoCoScore iterations of randomized hyperparameter 5-fold cross-validation using
    the given random_seed.
    """
)

parser.add_argument("--ft_threads", type=int, default=1)
args = parser.parse_args()
ft_threads = args.ft_threads

dataset_path = '/home/projects/jensenlab/people/nutmek/alex_data/new_split/sentences_dataset_bert_9606_-26_alex_sampled_train/dataset_9606_-26_train_sampled_alex.tsv.gz'
ft_dataset_path = '/home/projects/jensenlab/people/nutmek/alex_data/new_split/sentences_dataset_bert_9606_-26_alex_sampled_train/sentences_labels_9606_-26.txt'
ft_params = {'-dim': 300, '-epoch': 5, '-lr': 0.05}
fasttext_path = '/home/projects/jensenlab/people/nutmek/fastText/fasttext'
pretrained_embeddings = '/home/projects/jensenlab/people/nutmek/alex_data/fasttext_test/' \
                        'fasttext_sg_masked_dim_300_epoch_5_lr_0.05_minn_3_maxn_6_ws_5.vec'
tmp_model_path = '/home/projects/jensenlab/people/nutmek/alex_data/new_split/sentences_dataset_bert_9606_-26_alex_sampled_train/tmpmodel'
sentences_path = '/home/projects/jensenlab/people/nutmek/alex_data/new_split/sentences_dataset_bert_9606_-26_alex_sampled_train/sentence_9606_-26.txt'
prob_path = '/home/projects/jensenlab/people/nutmek/alex_data/new_split/sentences_dataset_bert_9606_-26_alex_sampled_train/probabilities.txt.gz'
results_file = 'results_9606_-26.pickle'
metric = 'roc_auc_score'
n_fold = 3
cocoscore_params = {'document_weight': 0.0, 'paragraph_weight': 0.0,
                    'sentence_weight': 1.0, 'weighting_exponent': 0.65,
                    'score_cutoff': 0.0}


df = pd.read_csv(dataset_path, sep='\t', compression='infer', header=None, index_col=None)
df.columns = ['pmid', 'paragraph', 'sentence', 'entity1', 'entity2', 'text', 'class', 'distance', 'pairs', 'key']
del df['pairs']
del df['key']

ft_df = pd.read_csv(ft_dataset_path, sep='\t', compression='infer', header=None, index_col=None)
sentence_df = pd.read_csv(sentences_path, sep='\t', compression='infer', header=None, index_col=None)

kf = KFold(n_splits=n_fold)

params_score = list()
for train_index, val_index in kf.split(df):
    train_df, ft_train_df = df.iloc[train_index], ft_df.iloc[train_index]
    val_df, ft_val_df = df.iloc[val_index], sentence_df.iloc[val_index]

    _, tmp_train_path = tempfile.mkstemp(text=True, suffix='.txt')
    with open(tmp_train_path, 'wt') as train_ft_out:
        ft_train_df.to_csv(train_ft_out, sep='\t', header=False, index=False)

    _, tmp_val_path = tempfile.mkstemp(text=True, suffix='.txt')
    with open(tmp_val_path, 'wt') as val_ft_out:
        ft_val_df.to_csv(val_ft_out, sep='\t', header=False, index=False)

    model_file = fasttext_fit(tmp_train_path, ft_params, fasttext_path, thread=ft_threads,
                              compress_model=True,
                              model_path=tmp_model_path,
                              pretrained_vectors_path=pretrained_embeddings)

    fth.fasttext_predict(model_file, tmp_val_path, fasttext_path, prob_path)
    probabilities = fth.load_fasttext_class_probabilities(prob_path)
    val_df = val_df.assign(predicted=probabilities)

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
                                         **cocoscore_params
                                         )
    val_performance = _compute_metric(val_score_dict, val_df, warn=False,
                                      metric=metric)
    params_score.append(val_performance)


with open(results_file, 'wb') as f:
    pickle.dump(params_score, f)
