# stellar-finetuning


## What is finetuning?

Fine-tuning refers to the technique of using deep pre-trained models for downstream tasks without gigantic datasets or massive GPU compute. It has become quite popular in recent times due to a lot of attractive features:
1. It allows users to leverage the information encoded in large language models without training them from scracth
2. Developers can get pretty good performance on their own datasets with minimal training. This is especially attractive for independent researchers and developers who may not have access to large GPU/TPU clusters
3. Quickly deploy models for low-impact use cases

## What will this repository teach me?

In this repository, we will go over how to fine-tune a language model. Speicifically, we will go over the following things:
1. Download a dataset (optionally, a specific configuration), model (we use bert-base-uncased)and its associated tokenizer from huggingface
2. Create training, validation and testing dataloaders using the encoder from the tokenizer
3. Define a parameterized function to freeze layers
4. Build a new model on top of the frozen (partially or complete) model 
5. Train and validate the model from step 4 on the training and validation dataloaders from step 2


  
