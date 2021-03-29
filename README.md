# stellar-finetuning


## What is finetuning?

Fine-tuning refers to the technique of using deep pre-trained models for downstream tasks without gigantic datasets or massive GPU compute. The idea behind fine-tuning is simple and elegant. Large language-models(LMs) are typically very deep and trained on massive datasets like C4 or WikiText. Like in other deep networks, it has been seen that lower levels of these networks learn simple semantics like part-of-speech and higher layers learn more advanced semantics like phrases and grammatical connections. The lower level semantics don't change very much for the language, but **small changes** occur in the advaned semantics like stylistic differences between authors, etc. By leveraging the lower-level semantics, these LMs can be used for a variety of downstream NLP tasks since they learn such a rich represnetation of the elements of the language. It has become quite popular in recent times due to a lot of attractive features:
1. It allows users to leverage the information encoded in large language models without training them from scratch.
2. Developers can get pretty good performance on their own datasets with minimal training. This is especially attractive for independent researchers and developers who may not have access to large GPU/TPU clusters
3. Quickly deploy models for low-impact use cases

## What will this repository teach me?

In this repository, we will go over how to fine-tune a language model. Speicifically, we will go over the following things:
1. Download a dataset (optionally, a specific configuration), model (we use bert-base-uncased)and its associated tokenizer from huggingface
2. Create training, validation and testing dataloaders using the encoder from the tokenizer
3. Define a parameterized function to freeze layers of a pre-trained model
4. Build a new model on top of the frozen (partially or complete) model 
5. Train and validate the model from step 4 on the training and validation dataloaders from step 2

## Let's get into it
### Get the data
Download dataset and an optional configuration from [huggingface datasets](https://huggingface.co/datasets):
1. Get a list of all datasets and pick one from the list. This reduces chances of users entering the wrong dataset name![image](https://user-images.githubusercontent.com/50837285/112832480-607e8b00-9063-11eb-995a-4249995745d8.png)
2. Some datasets require an optional configuration specified (eg. glue). Get the config name if such a dataset is picked ![image](https://user-images.githubusercontent.com/50837285/112831813-75a6ea00-9062-11eb-91c3-830eff1b2c3d.png)
3. Download the dataset and create the training, validation and test splits from the data NOTE: Not all datasets have the same splits. Check what splits are available and then assign or re-split them as required. Here's a [quick primer](https://huggingface.co/docs/datasets/splits.html) on how to split datasets in huggingface ![image](https://user-images.githubusercontent.com/50837285/112832213-febe2100-9062-11eb-9b32-d2049c634558.png)
4. Convert the data into the format required by the base_model that we will be using.

### Get the model
Now that we have the data, we need to find a model to fine-tune. You can pick any model of your choice, but my notebook uses ['bert-base-uncased'](https://huggingface.co/bert-base-uncased)
1. Get the model of your choice using the transformers package
2. Visualize it using the summary function from torchinfo 
![image](https://user-images.githubusercontent.com/50837285/112833370-a1c36a80-9064-11eb-8be5-507675ce091d.png)

### Define the classifier
Now that we have the base model, let's define the classifier architecture to use on top of this base model. I'm using a simple fully-connected classifier with a softmax over the output and ReLU activation. The arguments to this class' initializer are as follows:
- base_model (frozen or unfrozen)
- num of classes to predict
![image](https://user-images.githubusercontent.com/50837285/112834534-1a76f680-9066-11eb-8e2a-6e1ab4394351.png)
Let's take a look at what is happening in the forward pass of this architecture:
- The input tokens are passed through the frozen (partial or complete) base model
- The outputs from the model are then passed to the classifier which produces an output with dim=*num_classes* 

**Note: You might have to update your classifier architecture depending on the output from your base model**

### Let's freeze some layers
Now that we have everything in 
