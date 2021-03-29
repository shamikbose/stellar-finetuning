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
4. Convert the data into the format required by the base_model that we will be using
  - We define a function called encode() for this purpose
  - Next, we use the map() function of datasets to apply the same transformations to the train, test and validation datasets.
  - Next, we convert the transformed datasets into a PyTorch dataloader for easy consumption by our model

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

**Note: You might have to update your classifier architecture depending on the output from your base model and your specific use-case**

### Let's freeze some layers
What does it mean to freeze a layer in a model? During backpropagation, layer weights and biases are learnt as the model trains on the training set. When we freeze a layer, we are instructing the model to not update weights for these layers during training. During evaluation, the model is put into *eval* mode meaning all its weights are frozen during that phase. During training, we put the put the model into *train* mode, meaning that all the trainable parameters are unfrozen and can be re-trained.
Now that we have everything in place, we need to define a function to freeze some of the layers of our base_model that we downloaded earlier. 

![image](https://user-images.githubusercontent.com/50837285/112853454-dd692f00-907a-11eb-8c9f-7a2162302bf4.png)

The function freeze_layer() takes two inputs:
- **model** - this is the model for which we want to freeze the layers
- **freeze_layer_count** - as the name suggests, this tells the function how many layers are to be frozen. Set this to -1 if you want to train all layers except the embeddings learned by the model. If you want to freeze the first 3 layers, this value should be 3. Layers are 0-indexed

In PyTorch, every layer comes with a set of parameters. One of these parameters is *requires_grad*, which indicates to the optimizer that a layer's parameters need to be updated. To freeze a layer, we simply set this parameter to False. Let's run summary on this version of the model and see what it looks like

![image](https://user-images.githubusercontent.com/50837285/112853583-fffb4800-907a-11eb-8127-b9ff1be3bf25.png)

As we can see, the number of trainable parameters in the model has been halved from earlier, thanks to 5 of its 12 encoder layers being frozen. 

## Results

The following results were achieved with the following dataset and settings:
**dataset**: glue/mrpc
**epochs**: 10
**learning_rate**: 1e-4 with 0.01 weight decay
**batch_size**: 256

![image](https://user-images.githubusercontent.com/50837285/112876793-edd9d380-9093-11eb-8a0e-e79823541989.png)

