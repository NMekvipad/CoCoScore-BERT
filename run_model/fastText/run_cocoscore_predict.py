import argparse
import pickle
import pandas as pd
import tempfile
import sys
import gzip
import cocoscore.ml.fasttext_helpers as fth
from cocoscore.ml.fasttext_helpers import fasttext_fit
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

repeats = 3
ft_test_path = '/home/projects/jensenlab/people/nutmek/alex_data/new_split/' \
               'final_train_test/sentence_9606_-26_test_2.txt'
test_path = '/home/projects/jensenlab/people/nutmek/alex_data/new_split/' \
            'sentences_dataset_9606_-26_test_10k_2_alex/dataset_9606_-26_test_10k_2_alex.tsv.gz'
fasttext_path = '/home/projects/jensenlab/people/nutmek/fastText/fasttext'
tmp_model_path = '/home/projects/jensenlab/people/nutmek/alex_data/new_split/' \
                 'final_train_test/tmpmodel'
prob_path = '/home/projects/jensenlab/people/nutmek/alex_data/new_split/' \
            'final_train_test/probabilities.txt.gz'
results_file = 'results_9606_-26_final_test.pickle'
metric = 'roc_auc_score'

cocoscore_params = {'document_weight': 0.0, 'paragraph_weight': 0.0,
                    'sentence_weight': 1.0, 'weighting_exponent': 0.65,
                    'score_cutoff': 0.0}


test_df = pd.read_csv(test_path, sep='\t', compression='infer', header=None, index_col=None)
test_df.columns = ['pmid', 'paragraph', 'sentence', 'entity1', 'entity2', 'text', 'class', 'distance', 'pairs', 'key']
del test_df['key']
del test_df['pairs']

perfomance = list()
for i in range(repeats):
    fth.fasttext_predict(tmp_model_path + str(i) + '.ftz', ft_test_path, fasttext_path, prob_path)
    probabilities = fth.load_fasttext_class_probabilities(prob_path)
    test_df = test_df.assign(predicted=probabilities)

    _, tmp_file_path = tempfile.mkstemp(text=True, suffix='.gz')
    with gzip.open(tmp_file_path, 'wt') as test_out:
        test_df.to_csv(test_out, sep='\t', header=False, index=False,
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
    val_performance = _compute_metric(val_score_dict, test_df, warn=False, metric=metric)
    test_df.to_csv('ft_test_pred' + str(i) + '.tsv', sep='\t', header=False, index=False)
    perfomance.append(val_performance)

with open(results_file, 'wb') as f:
    pickle.dump(perfomance, f)

