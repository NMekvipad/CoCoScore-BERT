test_fraction = 0.2
root_dir = '/home/projects/jensenlab/people/nutmek/working_area/'
//dataset1 = Channel.fromPath(root_dir + 'sentences_dataset_bert_9606_-26/dataset_swapped_9606_-26.tsv.gz')
dataset2 = Channel.fromPath(root_dir + 'sentences_dataset_bert_9606_-25/dataset_swapped_9606_-25.tsv.gz')
//comention_output_dir1 = 'sentences_dataset_bert_9606_-26_split/'
comention_output_dir2 = 'sentences_dataset_bert_9606_-25_split/'

// Split each dataset into training and test set
//process train_test_splitting1 {
//    publishDir comention_output_dir1, mode: 'copy'
//    cpus 1
//    memory { 60.GB * task.attempt }
//    time '24 h'

//    input:
//    val test_fraction
//    file dataset1

//    output:
//    file "*_train.tsv.gz" into train_datasets
//    file "*_test.tsv.gz" into test_datasets

//    """
//    /home/projects/jensenlab/people/nutmek/bin/train_test_spliting.py $dataset1 $test_fraction
//    """
//}

process train_test_splitting2 {
    publishDir comention_output_dir2, mode: 'copy'
    cpus 1
    memory { 60.GB * task.attempt }
    time '24 h'

    input:
    val test_fraction
    file dataset2

    output:
    file "*_train.tsv.gz" into train_datasets2
    file "*_test.tsv.gz" into test_datasets2

    """
    /home/projects/jensenlab/people/nutmek/bin/train_test_spliting.py $dataset2 $test_fraction
    """
}
