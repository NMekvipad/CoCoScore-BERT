// First step of the pipeline that extracts co-mentions for each entity pair of interest and
// trains fastText word embeddings after masking entity matches in the corpus

// Init parameters
params.testrun = false
params.binding = false

comention_levels = Channel.from( 'sentence' )
comention_output_dir = 'sentences_dataset_bert_9606_-26/'
root_dir_corpus = '/home/projects/jensenlab/people/nutmek/corpus_data/'
root_dir_gold_pair = '/home/projects/jensenlab/people/nutmek/gold_data/'
pairs = Channel.from( [[9606, -26]] )
match_entities_awk = '{if (\$7 == "9606" || \$7 == "-26" ) print \$0}'
//pairs = Channel.from( [[9606, -25]] )
//match_entities_awk = '{if (\$7 == "9606" || \$7 == "-25" ) print \$0}'

if (params.testrun) {

    pair_gold_standard_files = [
    [9606, -26]: file(root_dir_corpus + 'testdata/9606_-26_gold.tsv.gz'),
    [9606, -25]: file(root_dir_corpus + 'testdata/9606_-25_gold.tsv.gz')]
    expected_datasets_dir = Channel.fromPath(root_dir_corpus + 'testdata/expected_datasets/' )

    // corpus and tagger output files used as input to the pipeline
    all_matches_file = Channel.fromPath(root_dir_corpus +  'testdata/corpus_matches.tsv.gz' )
    corpus_file = Channel.fromPath(root_dir_corpus +  'testdata/corpus.tsv.gz' )
    segments_file = Channel.fromPath(root_dir_corpus +  'testdata/corpus_segments.tsv.gz' )
    entity_file = Channel.fromPath(root_dir_corpus +  'testdata/corpus_entities.tsv.gz' )
    names_file = Channel.fromPath(root_dir_corpus +  'testdata/corpus_names.tsv.gz' )
    preferred_names_file = Channel.fromPath(root_dir_corpus +  'testdata/corpus_names_preferred.tsv.gz' )
    entity_file1 = Channel.fromPath(root_dir_corpus +  'testdata/corpus_entities.tsv.gz' )
    names_file1 = Channel.fromPath(root_dir_corpus +  'testdata/corpus_names.tsv.gz' )
    preferred_names_file1 = Channel.fromPath(root_dir_corpus +  'testdata/corpus_names_preferred.tsv.gz' )
    comention_file_count = 1
    corpus_file_count = 1
    test_fraction = 0.5
} else {
    pair_gold_standard_files = [
    [9606, -26]: file(root_dir_gold_pair + '01-disease-gene/ghr.tsv.gz'),
    [9606, -25]: file(root_dir_gold_pair + '03-gene-tissue/uniprotkb_tissues.tsv.gz')]

    // corpus and tagger output files used as input to the pipeline
    document_output_dir = 'documents/'
    all_matches_file = Channel.fromPath(root_dir_corpus +  'corpus_matches.tsv.gz' )
    corpus_file= Channel.fromPath(root_dir_corpus +  'corpus.tsv.gz' )
    segments_file = Channel.fromPath(root_dir_corpus +  'corpus_segments.tsv.gz' )
    entity_file = Channel.fromPath(root_dir_corpus +  'all_entities.tsv.gz' )
    names_file = Channel.fromPath(root_dir_corpus +  'all_names_cleaned.tsv.gz' )
    preferred_names_file = Channel.fromPath(root_dir_corpus +  'all_names_preferred.tsv.gz' )
    entity_file1 = file(root_dir_corpus + 'all_entities.tsv.gz')
    names_file1 = file(root_dir_corpus + 'all_names_cleaned.tsv.gz')
    preferred_names_file1 = file(root_dir_corpus + 'all_names_preferred.tsv.gz')
    // TODO replace repeated entity, names, preferred names files with single channel
    comention_file_count = 30
    corpus_file_count = 15
    test_fraction = 0.2
}
comention_output_dir_path = file(comention_output_dir)


/*
 *
 *  Processes needed to train fastText embeddings
 *
 */

// Keep only tagger matches for entity types of interest
process filter_matches {
    executor 'local'

    input:
    file all_matches_file
    val match_entities_awk

    output:
    file 'matches_filtered.tsv.gz' into entities_matches_file_comentions

    """
    gzip -cd $all_matches_file | awk -F"\\t" '$match_entities_awk' | gzip > matches_filtered.tsv.gz
    """
}

/*
 *
 *  Processes needed to extract co-mentions and text fragments
 *
 */
segments_file.into { sentence_segments_file; segments_file_paragraph; segments_file_documents }

// Merge sentence-level segments, as produced by tagger, into paragraph-level segments
process create_paragraph_segments {
    cpus 1
    executor 'local'

    input:
    file segments_file from segments_file_paragraph

    output:
    file "paragraph_segments.tsv.gz" into paragraph_segments_file

    """
    /home/projects/jensenlab/people/nutmek/bin/create_paragraph_segments_file.py $segments_file | gzip > paragraph_segments.tsv.gz
    """
}

// Merge sentence-level segments, as produced by tagger, into document-level segments
process create_document_segments {
    cpus 1
    executor 'local'

    input:
    file segments_file from segments_file_documents

    output:
    file "document_segments.tsv.gz" into document_segments_file

    """
    /home/projects/jensenlab/people/nutmek/bin/create_document_segments_file.py $segments_file | gzip > document_segments.tsv.gz
    """
}

// Split matches file into smaller chunks
process split_comention_matches {
    cpus 1
    executor 'local'

    input:
    file entities_matches_file_comentions
    val comention_file_count

    output:
    file('matches_part_*.tsv.gz') into matches_split mode flatten

    """
    /home/projects/jensenlab/people/nutmek/bin/split_matches.py $entities_matches_file_comentions --target_file_count $comention_file_count --output_file_prefix matches_part_
    """
}

// Filter matches file, as produced by tagger, to co-mentions at sentence-/paragraph-/document-level for
// each pair of entities (basically just filtering for entities in match file that is in gold standard and form pair
// and throw filtered match file as comentioned match file to next step)
process filter_comentions_from_matches {
    cpus 1
    memory { 16.GB * task.attempt }
    time { 24.hour * task.attempt }

    input:
    file(matches_file) from matches_split
    each file(entity_file)
	each file(names_file)
	each file(preferred_names_file)
    each entity_pair from pairs
    each comention_level from comention_levels

    output:
    set first_entity, second_entity, comention_level, file('comentions_*.tsv.gz') into filtered_comentions_matches_split_single

    script:
    first_entity = entity_pair[0]
    second_entity = entity_pair[1]
    gold_file = pair_gold_standard_files[entity_pair]
    """
    export PYTHONIOENCODING='UTF-8'
    /home/projects/jensenlab/people/nutmek/bin/prepare_matches.py $matches_file $first_entity $second_entity $comention_level $entity_file $names_file $preferred_names_file $gold_file | gzip > comentions_$matches_file
    """
}

// Extract text fragments containing sentence- and paragraph-level co-mentions.
// For document-level co-mentions, a file in the same format as for sentence-level and
// paragraph-level co-mentions is produced but no text fragments are extracted
// since extracting full-text fragments would result in a huge file.
process extract_text {
    cpus 1
    memory { 16.GB * task.attempt }
    time { 24.hour * task.attempt }

    input:
    each file(corpus_file_comentions) from corpus_file
    each file(sentence_segments_file) from sentence_segments_file
    each file(paragraph_segments_file) from paragraph_segments_file
    each file(document_segments_file) from document_segments_file
    set first_entity, second_entity, comention_level, file(matches_file) from filtered_comentions_matches_split_single

    output:
    set first_entity, second_entity, comention_level, file('text_*.tsv.gz') into filtered_comentions_text_matches_split_single

    script:
    if(comention_level == 'sentence') {
        segments_file =  sentence_segments_file
    } else if (comention_level == 'paragraph') {
        segments_file = paragraph_segments_file
    } else {
        segments_file = document_segments_file
    }
    """
    export PYTHONIOENCODING='UTF-8'

    /home/projects/jensenlab/people/nutmek/bin/extract_text.py $corpus_file_comentions $segments_file \
    $matches_file $comention_level | gzip > text_$matches_file
    """
}

/*
 *
 * Processes needed to generate the dataset with labels and split into train/test set.
 *
 */
// Assign 0/1 labels to each pair based on gold standard
process generate_dataset_split {
	cpus 21
    memory { 32.GB * task.attempt }
    time { 24.hour * task.attempt }

	input:
	each file(entity_file) from entity_file1
	each file(names_file) from names_file1
	each file(preferred_names_file) from preferred_names_file1
	set first_entity, second_entity, comention_level, file(matches_file) from filtered_comentions_text_matches_split_single

	output:
	set first_entity, second_entity, comention_level, file('dataset_*.tsv.gz') into dataset_split_single

    script:
    gold_file = pair_gold_standard_files[[first_entity, second_entity]]
    if (params.binding) {
        gray_list_arg = '--gray_list_file ' + pair_gray_list_files[[first_entity, second_entity]]
    } else {
        gray_list_arg = ''
    }
    outfile = 'dataset_' + comention_level + '_' +matches_file
	"""
	export PYTHONIOENCODING='UTF-8'
	/home/projects/jensenlab/people/nutmek/bin/generate_dataset.py $entity_file $names_file $preferred_names_file $gold_file $matches_file $first_entity $second_entity $comention_level --allow_multiple_pairs --allow_non_gold_standard_entities $gray_list_arg | gzip > $outfile
	"""
}

dataset_split_combined = dataset_split_single.groupTuple(by: [0, 1])

// Combine co-mentions of all levels and all parts into one file per entity pair.
process collect_dataset_splits {
    //publishDir comention_output_dir, mode: 'copy'
    executor 'local'
    cpus 1
    tag { first_entity + '_' + second_entity }

    input:
    set first_entity, second_entity, comention_levels, file(dataset_split_files) from dataset_split_combined

    output:
    set first_entity, second_entity, file('dataset_*.tsv.gz') into datasets_unswapped

    script:
    output_file = "dataset_" + first_entity + "_" + second_entity + ".tsv.gz"
    """
    cat $dataset_split_files > $output_file
    """
}

datasets_unswapped.into{ datasets_unswapped_positive; datasets_unswapped_negative; datasets_unswapped_to_swap }

process filter_positive {
    executor 'local'
    cpus 1
    tag { first_entity + '_' + second_entity + '_positive' }

    input:
    set first_entity, second_entity, file(dataset_unswapped_positive) from datasets_unswapped_positive

    output:
    set first_entity, second_entity, file('dataset_positive_*.tsv.gz') into dataset_positive_pack

    script:
    output_file = "dataset_positive_" + first_entity + "_" + second_entity + ".tsv.gz"
    """
    gzip -cd $dataset_unswapped_positive | awk -F"\\t" '{if (\$8 == "1") print \$0}' | gzip > $output_file
    """

}

process extract_entities_positive {
    executor 'local'
    cpus 1
    memory { 8.GB * task.attempt }
    time { 48.hour * task.attempt }

    tag { first_entity + '_' + second_entity + '_positive_ent' }

    input:
    set first_entity, second_entity, file(dataset_positive) from dataset_positive_pack

    output:
    set first_entity, second_entity, file('entities_positive_*.tsv.gz') into entities_positive_pack

    script:
    output_file = "entities_positive_" + first_entity + "_" + second_entity + ".tsv.gz"
    """
    /home/projects/jensenlab/people/nutmek/bin/extract_entities.py $dataset_positive | gzip > $output_file
    """
}

entities_positive_pack.into{ entities_positive_pack_one; entities_positive_pack_two }

process extract_first_entities {
    executor 'local'
    cpus 1
    tag { first_entity + '_' + second_entity + '_positive_ent_first' }

    input:
    set first_entity, second_entity, file(entities_positive) from entities_positive_pack_one

    output:
    set first_entity, file('first_entities_positive_*.tsv') into first_entities_positive_pack

    script:
    output_file = "first_entities_positive_" + first_entity + "_" + second_entity + ".tsv"
    awk_query = '{if (\$7 == "' + first_entity + '") print $0}'
    """
    gzip -cd $entities_positive | awk -F"\\t" '$awk_query' > $output_file
    """
}

first_entities_positive_pack.into{ first_entities_positive; first_entities_positive_pool }

process extract_second_entities {
    executor 'local'
    cpus 1
    tag { first_entity + '_' + second_entity + '_positive_ent_second' }

    input:
    set first_entity, second_entity, file(entities_positive) from entities_positive_pack_two

    output:
    set second_entity, file('second_entities_positive_*.tsv') into second_entities_positive_pack

    script:
    output_file = "second_entities_positive_" + first_entity + "_" + second_entity + ".tsv"
    awk_query = '{if (\$7 == "' + second_entity + '") print $0}'
    """
    gzip -cd $entities_positive | awk -F"\\t" '$awk_query' > $output_file
    """
}

second_entities_positive_pack.into{ second_entities_positive; second_entities_positive_pool }

process shuffle_first_entities {
    executor 'local'
    cpus 1
    memory { 8.GB * task.attempt }
    time { 48.hour * task.attempt }
    tag { first_entity + '_' + '_positive_ent' }

    input:
    set first_entity, file(first_positive) from first_entities_positive

    output:
    set first_entity, file('first_shuffled_*.tsv.gz') into shuffled_first_positive

    script:
    output_file = "first_shuffled_" + first_entity + ".tsv.gz"
    """
    awk -F"\\t" '{print \$6}' $first_positive | shuf | paste $first_positive /dev/stdin | gzip > $output_file
    """
}

process shuffle_second_entities {
    executor 'local'
    cpus 1
    memory { 8.GB * task.attempt }
    time { 48.hour * task.attempt }
    tag { second_entity + '_' + '_positive_ent' }

    input:
    set second_entity, file(second_positive) from second_entities_positive

    output:
    set second_entity, file('second_shuffled_*.tsv.gz') into shuffled_second_positive

    script:
    output_file = "second_shuffled_" + second_entity + ".tsv.gz"
    """
    awk -F"\\t" '{print \$6}' $second_positive | shuf | paste $second_positive /dev/stdin | gzip > $output_file
    """
}

process pool_positive_entities {
    executor 'local'
    cpus 1
    tag { 'positive_entities_pooling' }

    input:
    set first_entity, file(first_positive_shuffled) from shuffled_first_positive
    set second_entity, file(second_positive_shuffled) from shuffled_second_positive

    output:
    set first_entity, second_entity, file('positive_shuffled_*.tsv.gz') into shuffled_positive

    script:
    output_file = "positive_shuffled_" + first_entity + "_" + second_entity + ".tsv.gz"
    """
    cat $first_positive_shuffled $second_positive_shuffled > $output_file
    """
}

process filter_negative {
    executor 'local'
    cpus 1
    tag { first_entity + '_' + second_entity + '_negative' }

    input:
    set first_entity, second_entity, file(dataset_unswapped_negative) from datasets_unswapped_negative

    output:
    set first_entity, second_entity, file('dataset_negative_*.tsv.gz') into dataset_negative_pack

    script:
    output_file = "dataset_negative_" + first_entity + "_" + second_entity + ".tsv.gz"
    """
    gzip -cd $dataset_unswapped_negative | awk -F"\\t" '{if (\$8 == "0") print \$0}' | gzip > $output_file
    """
}

process extract_entities_negative {
    executor 'local'
    cpus 1
    memory { 8.GB * task.attempt }
    time { 48.hour * task.attempt }
    tag { first_entity + '_' + second_entity + '_negative_ent' }

    input:
    set first_entity, second_entity, file(dataset_negative) from dataset_negative_pack

    output:
    set first_entity, second_entity, file('entities_negative_*.tsv.gz') into entities_negative_pack

    script:
    output_file = "entities_negative_" + first_entity + "_" + second_entity + ".tsv.gz"
    """
    /home/projects/jensenlab/people/nutmek/bin/extract_entities.py $dataset_negative | gzip > $output_file
    """
}

entities_negative_pack.into{ entities_negative_pack_one; entities_negative_pack_two }

process extract_first_negative_entities {
    executor 'local'
    cpus 1
    tag { first_entity + '_' + second_entity + '_negative_ent_first' }

    input:
    set first_entity, second_entity, file(entities_negative) from entities_negative_pack_one

    output:
    set first_entity, file('first_entities_negative_*.tsv') into first_entities_negative_pack

    script:
    output_file = "first_entities_negative_" + first_entity + "_" + second_entity + ".tsv"
    awk_query = '{if (\$7 == ' + first_entity + ') print \$0}'
    """
    gzip -cd $entities_negative | awk -F"\\t" '$awk_query' > $output_file
    """
}

process extract_second_negative_entities {
    executor 'local'
    cpus 1
    tag { first_entity + '_' + second_entity + '_negative_ent_second' }

    input:
    set first_entity, second_entity, file(entities_negative) from entities_negative_pack_two

    output:
    set second_entity, file('second_entities_negative_*.tsv') into second_entities_negative_pack

    script:
    output_file = "second_entities_negative_" + first_entity + "_" + second_entity + ".tsv"
    awk_query = '{if (\$7 == "' + second_entity + '") print \$0}'
    """
    gzip -cd $entities_negative | awk -F"\\t" '$awk_query' > $output_file
    """
}


process sampling_first_entities{
    executor 'local'
    cpus 1
    memory { 8.GB * task.attempt }
    time { 48.hour * task.attempt }
    tag { first_entity + '_negative_ent' }

    input:
    set first_entity, file(first_positive_pool) from first_entities_positive_pool
    set first_entity, file(first_negative) from first_entities_negative_pack

    output:
    set first_entity, file('first_shuffled_negative_*.tsv.gz') into shuffled_first_negative

    script:
    output_file = "first_shuffled_negative_" + first_entity + ".tsv.gz"
    """
    /home/projects/jensenlab/people/nutmek/bin/sample_positive.sh  $first_positive_pool $first_negative $output_file
    """
}

process sampling_second_entities{
    executor 'local'
    cpus 1
    memory { 8.GB * task.attempt }
    time { 48.hour * task.attempt }
    tag { second_entity + '_negative_ent' }

    input:
    set second_entity, file(second_positive_pool) from second_entities_positive_pool
    set second_entity, file(second_negative) from second_entities_negative_pack

    output:
    set second_entity, file('second_shuffled_negative_*.tsv.gz') into shuffled_second_negative

    script:
    output_file = "second_shuffled_negative_" + second_entity + ".tsv.gz"
    """
    /home/projects/jensenlab/people/nutmek/bin/sample_positive.sh  $second_positive_pool $second_negative $output_file
    """
}

process pool_negative_entities {
    executor 'local'
    cpus 1
    tag { 'positive_entities_pooling' }

    input:
    set first_entity, file(first_negative_shuffled) from shuffled_first_negative
    set second_entity, file(second_negative_shuffled) from shuffled_second_negative

    output:
    set first_entity, second_entity, file('negative_shuffled_*.tsv.gz') into shuffled_negative

    script:
    output_file = "negative_shuffled_" + first_entity + "_" + second_entity + ".tsv.gz"
    """
    cat $first_negative_shuffled $second_negative_shuffled > $output_file
    """
}

process pool_entities {
    executor 'local'
    cpus 1
    tag { 'entities_pooling' }

    input:
    set first_entity, second_entity, file(shuffled_positive_entities) from shuffled_positive
    set first_entity, second_entity, file(shuffled_negative_entities) from shuffled_negative

    output:
    set first_entity, second_entity, file('shuffled_entities_*.tsv.gz') into shuffled_entities

    script:
    output_file = "shuffled_entities_" + first_entity + "_" + second_entity + ".tsv.gz"
    """
    cat $shuffled_positive_entities $shuffled_negative_entities > $output_file
    """
}

process sort_dataset {
    executor 'local'
    cpus 8
    tag { 'dataset_sorting' }

    input:
    set first_entity, second_entity, file(dataset) from datasets_unswapped_to_swap

    output:
    set first_entity, second_entity, file('sorted_dataset_*.tsv') into sorted_dataset

    script:
    output_file = "sorted_dataset_" + first_entity + "_" + second_entity + ".tsv"
    """
    gzip -cd $dataset | sort -n -k 1,1 -k 2,2 -k 3,3 -k 4,4 --parallel=${task.cpus} > $output_file
    """
}

process sort_entities {
    executor 'local'
    cpus 8
    tag { 'entities_sorting' }

    input:
    set first_entity, second_entity, file(entities) from shuffled_entities

    output:
    set first_entity, second_entity, file('sorted_entities_*.tsv') into sorted_entities

    script:
    output_file = "sorted_entities_" + first_entity + "_" + second_entity + ".tsv"
    """
    gzip -cd $entities | sort -n -k 1,1 -k 2,2 -k 3,3 -k 4,4 -k 5,5 --parallel=${task.cpus} > $output_file
    """
}

process condense_entities {
    executor 'local'
    cpus 1
    tag { 'entities_condensing' }

    input:
    set first_entity, second_entity, file(entities) from sorted_entities

    output:
    set first_entity, second_entity, file('extracted_entities_*.tsv') into extracted_entities

    script:
    output_file = "extracted_entities_" + first_entity + "_" + second_entity + ".tsv"
    """
    /home/projects/jensenlab/people/nutmek/bin/collect_entities.py $entities > $output_file
    """
}

process concat_dataset_entities {
    executor 'local'
    cpus 1
    tag { 'concat_entities_column' }

    input:
    set first_entity, second_entity, file(entities) from extracted_entities
    set first_entity, second_entity, file(dataset) from sorted_dataset

    output:
    set first_entity, second_entity, file("dataset_w_swap_info_*.tsv.gz") into dataset_w_swap_info

    script:
    output_file = "dataset_w_swap_info_" + first_entity + "_" + second_entity + ".tsv.gz"
    """
    awk -F"\\t" '{print \$5}' $entities | paste $dataset /dev/stdin | gzip > $output_file
    """
}

process swap_entities_in_text {
    publishDir comention_output_dir, mode: 'copy'
    executor 'local'
    cpus 1
    tag { 'swapping_entities_in_text' }

    input:
    set first_entity, second_entity, file(dataset_w_info) from dataset_w_swap_info

    output:
    file("dataset_swapped_*.tsv.gz") into datasets

    script:
    output_file = "dataset_swapped_" + first_entity + "_" + second_entity + ".tsv.gz"
    """
    /home/projects/jensenlab/people/nutmek/bin/swap_name.py $dataset_w_info | gzip > $output_file
    """
}

// Split each dataset into training and test set
//process train_test_splitting {
//    publishDir comention_output_dir, mode: 'copy'
//    cpus 1
//    memory { 60.GB * task.attempt }
//    time '24 h'

//    input:
//    val test_fraction
//    file dataset from datasets

//    output:
//    file "*_train.tsv.gz" into train_datasets
//    file "*_test.tsv.gz" into test_datasets
//    val 'something' into x


//    """
//    train_test_splitting.py $dataset $test_fraction
//    """
//}


