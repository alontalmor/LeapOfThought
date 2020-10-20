local train_size = 10000;
local batch_size = 8;
local gradient_accumulation_batch_size = 2;
local num_epochs = 4;
local learning_rate = 5e-7;
local weight_decay = 0.1;
local warmup_ratio = 0.06;
local transformer_model = "roberta-base";
local cuda_device = 0;

{
  "dataset_reader": {
    "type": "transformer_binary_qa",
    "sample": train_size,
    "pretrained_model": transformer_model,
    "max_pieces": 256
  },
  "validation_dataset_reader": {
    "type": "transformer_binary_qa",
    "sample": -1,
    "pretrained_model": transformer_model,
    "max_pieces": 256
  },
  //"datasets_for_vocab_creation": [],
  "train_data_path": "s3://olmpics/challenge/commonsense_knowledge_train.jsonl.gz",
  "validation_data_path": "s3://olmpics/challenge/commonsense_knowledge_dev.jsonl.gz",

  "model": {
    "type": "transformer_binary_qa",
    "pretrained_model": transformer_model
  },
  "data_loader": {
      "batch_sampler": {
        "type": "bucket",
        "batch_size": batch_size
      }
  },
  "trainer": {
    "optimizer": {
      "type": "huggingface_adamw",
      "weight_decay": weight_decay,
      "betas": [0.9, 0.98],
      "parameter_groups": [[["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}]],
      "lr": learning_rate
    },
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "num_epochs": num_epochs,
      "cut_frac": warmup_ratio,
      "num_steps_per_epoch": std.ceil(train_size / batch_size /  gradient_accumulation_batch_size),
    },
    "validation_metric": "+EM",
     "checkpointer": {
        "num_serialized_models_to_keep": 1
    },
    //"should_log_learning_rate": true,
    "num_gradient_accumulation_steps": gradient_accumulation_batch_size,
    // "grad_clipping": 1.0,
    "num_epochs": num_epochs,
    "cuda_device": cuda_device
  }
}