{
    "dataset_reader": {
        "type": "transformer_binary_qa",
        "sample": -1,
        "combine_input_fields": false,
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": false
            }
        }
    },
    "train_data_path": "https://aigame.s3-us-west-2.amazonaws.com/data/hypernyms/hypernyms_training_mix_train.jsonl.gz",
    "validation_data_path": "https://aigame.s3-us-west-2.amazonaws.com/data/hypernyms/hypernyms_training_mix_dev.jsonl.gz",
    "model": {
        "type": "esim_binary_qa",
        "dropout": 0.5,
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    //"pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz",
                    "embedding_dim": 50,
                    "trainable": true
                }
            }
        },
        "encoder": {
            "type": "lstm",
            "input_size": 50,
            "hidden_size": 300,
            "num_layers": 1,
            "bidirectional": true
        },
        "matrix_attention": {
            "type": "dot_product"
        },
        "projection_feedforward": {
            "input_dim": 2400,
            "hidden_dims": 300,
            "num_layers": 1,
            "activations": "relu"
        },
        "inference_encoder": {
            "type": "lstm",
            "input_size": 300,
            "hidden_size": 300,
            "num_layers": 1,
            "bidirectional": true
        },
        "output_feedforward": {
            "input_dim": 2400,
            "num_layers": 1,
            "hidden_dims": 300,
            "activations": "relu",
            "dropout": 0.5
        },
        "output_logit": {
            "input_dim": 300,
            "num_layers": 1,
            "hidden_dims": 2,
            "activations": "linear"
        },
        "initializer": {
          "regexes": [
            [".*linear_layers.*weight", {"type": "xavier_uniform"}],
            [".*linear_layers.*bias", {"type": "zero"}],
            [".*weight_ih.*", {"type": "xavier_uniform"}],
            [".*weight_hh.*", {"type": "orthogonal"}],
            [".*bias_ih.*", {"type": "zero"}],
            [".*bias_hh.*", {"type": "lstm_hidden_bias"}]
          ]
        }
    },
    "data_loader": {
          "batch_sampler": {
            "type": "bucket",
            "batch_size": 16
          }
    },
    //"data_loader": {
    //  "batch_size": 16,
    //  "shuffle": false
    //},
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 0.0004
        },
        "checkpointer": {
            "num_serialized_models_to_keep": 2,
        },
        "validation_metric": "+EM",
        "num_epochs": 75,
        "grad_norm": 10.0,
        "patience": 20,
        "cuda_device": 0,
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "mode": "max",
            "patience": 5
        }
    }
}