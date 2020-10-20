# LeapOfThought

This work was performed at The Allen Institute of Artificial Intelligence.

This project is constantly being improved. Contributions, comments and suggestions are welcome!

## Artisets (Artificial Datasets)

| Experiment | Subset | Data   
| :----- | :----- | :-----:|  
|  Implicit Knowledge of Taxonomic Relations | Training | [train](https://aigame.s3-us-west-2.amazonaws.com/data/hypernyms/hypernyms_training_mix_short_train.jsonl.gz), [dev](https://aigame.s3-us-west-2.amazonaws.com/data/hypernyms/hypernyms_training_mix_short_dev.jsonl.gz) 
|  Implicit Knowledge of Taxonomic Relations | Hypothesis-only | [dev](https://aigame.s3-us-west-2.amazonaws.com/data/hypernyms/hypernyms_statement_only_short_neg_hypernym_rule_dev.jsonl.gz), [test](https://aigame.s3-us-west-2.amazonaws.com/data/hypernyms/hypernyms_statement_only_short_neg_hypernym_rule_test.jsonl.gz) 
|  Implicit Knowledge of Taxonomic Relations | ExplicitReasoning | [dev](https://aigame.s3-us-west-2.amazonaws.com/data/hypernyms/hypernyms_explicit_only_short_neg_hypernym_rule_dev.jsonl.gz), [test](https://aigame.s3-us-west-2.amazonaws.com/data/hypernyms/hypernyms_explicit_only_short_neg_hypernym_rule_test.jsonl.gz) 
|  Implicit Knowledge of Taxonomic Relations | ImplicitReasoning | [dev](https://aigame.s3-us-west-2.amazonaws.com/data/hypernyms/hypernyms_implicit_only_short_neg_hypernym_rule_dev.jsonl.gz), [test](https://aigame.s3-us-west-2.amazonaws.com/data/hypernyms/hypernyms_implicit_only_short_neg_hypernym_rule_test.jsonl.gz) 
|  Counting over Implicit Facts | Training | [train](https://aigame.s3-us-west-2.amazonaws.com/data/counting/counting_training_mix_train.jsonl.gz), [dev](https://aigame.s3-us-west-2.amazonaws.com/data/counting/counting_training_mix_dev.jsonl.gz) 
|  Counting over Implicit Facts | Hypothesis-only | [dev](https://aigame.s3-us-west-2.amazonaws.com/data/counting/counting_statement_only_between_one_and_full_counted_instances_dev.jsonl.gz), [test](https://aigame.s3-us-west-2.amazonaws.com/data/counting/counting_statement_only_between_one_and_full_counted_instances_test.jsonl.gz) 
|  Counting over Implicit Facts | Counting (1, K-1) | [dev](https://aigame.s3-us-west-2.amazonaws.com/data/counting/counting_real_fact_only_between_one_and_full_counted_instances_dev.jsonl.gz), [test](https://aigame.s3-us-west-2.amazonaws.com/data/counting/counting_real_fact_only_between_one_and_full_counted_instances_test.jsonl.gz) 
|  Counting over Implicit Facts | Counting K | [dev](https://aigame.s3-us-west-2.amazonaws.com/data/counting/counting_real_fact_only_count_reached_dev.jsonl.gz), [test](https://aigame.s3-us-west-2.amazonaws.com/data/counting/counting_real_fact_only_count_reached_test.jsonl.gz)

## Setup

### Setting up a virtual environment

1.  First, clone the repository:

    ```
    git clone https://github.com/alontalmor/LeapOfThought.git
    ```

2.  Change your directory to where you cloned the files:

    ```
    cd LeapOfThought
    ```

3.  Create a virtual environment with Python 3.6 or above:

    ```
    virtualenv venv --python=python3.7 (or python3.7 -m venv venv or conda create -n multiqa python=3.7)
    ```

4.  Activate the virtual environment. You will need to activate the venv environment in each terminal in which you want to use LeapOfThought.

    ```
    source venv/bin/activate (or source venv/bin/activate.csh or conda activate multiqa)
    ```
5.  Install the required dependencies:

    ```
    pip install -r requirements.txt
    ```

### Tests

You can test all artisets using pytest, or using pycharm tests directory (pytest-pycharm added):

`pytest tests`

### Data

The allennlp caching infra is used, so be sure to have enough disk space, and control the cache directory using ALLENNLP_CACHE_ROOT env variable.

## Create a new Artiset

   This will take create a new artiset python file and test based on the copy_from artiset, output is placed in artiset_module dir.

  `python LeapOfThought/run.py -c Hypernyms -o create_new_artiset --copy_from ExampleArtiset --artiset_module soft_reasoning`
  
## Build Artiset

   From please use:
   `python LeapOfThought/run.py --help`

   To recreate the hypernyms training data 
  `python LeapOfThought/run.py -c Hypernyms -o build_artificial_dataset -out s3://aigame/data/taxonomic_reasonings/taxonomic_reasonings.jsonl.gz -v trainin_mix`


## Training using AllenNLP

For training the counting experiment (hypernym experiment has similar hyperparams):
`python -m allennlp train LeapOfThought/allennlp_models/config/transformer_binary_qa.jsonnet -s YOUR_OUTPUT_DIR -o "{'data_loader': {'batch_sampler': {'batch_size': 4}}, 'dataset_reader': {'pretrained_model': 'roberta-large', 'sample': 35000}, 'model': {'pretrained_model': 'roberta-large'}, 'random_seed': 2, 'train_data_path': 'https://aigame.s3-us-west-2.amazonaws.com/data/counting/counting_training_mix_no_total_train.jsonl.gz', 'trainer': {'checkpointer': {'num_serialized_models_to_keep': 1}, 'cuda_device': GPUNUM, 'learning_rate_scheduler': {'cut_frac': 0.1, 'num_epochs': 4, 'num_steps_per_epoch': 729}, 'num_epochs': 4, 'num_gradient_accumulation_steps': 12, 'optimizer': {'lr': 1e-05, 'weight_decay': 0.1}}, 'validation_data_path': 'https://aigame.s3-us-west-2.amazonaws.com/data/counting/counting_training_mix_no_total_dev.jsonl.gz', 'validation_dataset_reader': {'pretrained_model': 'roberta-large'}}"  --include-package LeapOfThought.allennlp_models`

## Evaluating  using AllenNLP

Example for evaluating the Implicit Knowledge of Taxonomic Relations, ExplicitReasoning test set:
`python -m allennlp evaluate MY_TRAINED_MODEL_PATH.tar.gz https://aigame.s3-us-west-2.amazonaws.com/data/hypernyms/hypernyms_explicit_only_short_neg_hypernym_rule_test.jsonl.gz --output-file MY_OUTPUT_FILE_results.json -o "{'trainer': {'cuda_device': GPUNUM}, 'validation_data_loader': {'batch_sampler': {'batch_size': 20, 'type': 'bucket'}}}"  --cuda-device -1 --include-package LeapOfThought.allennlp_models`


## Other
A caching infra is used, so make sure to have enough disk space, and control the cache directory using `LEAPOFTHOUGHT_CACHE_ROOT` env variable.
see teachyourai/common/file_utils.py





