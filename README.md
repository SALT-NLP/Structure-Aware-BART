# Structure-Aware-BART
This repo contains codes for the following paper: 

*Jiaao Chen, Diyi Yang*:Structure-Aware Abstractive Conversation Summarization via Discourse and Action Graphs,  NAACL 2021

If you would like to refer to it, please cite the paper mentioned above. 


## Getting Started
These instructions will get you running the codes of Structure-Aware-Bart Conversation Summarization.

### Requirements
* Python 3.6 or higher
* Pytorch >= 1.3.0
* Pandas, Numpy, Pickle
* rouge=1.0.0 (https://github.com/pltrdy/rouge)
* transformers
* allennlp
* openie
* wandb


**Note that different versions of rouge or different rouge packages might result in different rouge scores.**
**For the transformers, we used the version released by Oct. 7 2020. The updated version might also result in different performances.**



### Install the transformers with S-BART

```
cd transformers

pip install --editable ./
```


### Downloading the data
Please download the dataset (including pre-processed graphs) and put them in the data folder [here](https://drive.google.com/file/d/1PisTpC13e0yOEfSwLGtq5JO1q4kDg_ta/view?usp=sharing)

### Pre-processing the data

The data folder you download from the above link already contains all the pre-processed files (including the extracted graphs) from SAMSum corpus.

#### Extract Discourse Graphs

Here we utilize the data and codes from [here](https://github.com/shizhouxing/DialogueDiscourseParsing) to pre-train a conversation discourse parser and use that parser to extract discourse graphs in the SAMSum dataset.


#### Extract Action Graphs

Please go through `./src/data/extract_actions.ipynb` to extract action graphs.

 
### Training models

These section contains instructions for training the conversation summarizationmodels.

The generated summaries on test set for baseline BART and the S-BART is in the `./src/baseline` and `./src/composit` folder.  (trained with seed 42)

The training logs from wandb for different seed (0,1,42) for S-BART is shown in `./src/Weights&Biases.pdf`


#### Training baseline BART model
Please run `./train_base.sh` to train the BART baseline models.


#### Training S-BART model
Please run `./train_multi_graph.sh` to train the S-BART model. 

### Evaluating models

Please follow the example jupyter notebook (`./src/eval.ipynb`) is provided for evaluating the model on test set. 





