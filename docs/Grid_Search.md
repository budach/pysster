# Class Grid\_Search - Documentation

The Grid\_Search class provides a simple way to execute a hyperparameter tuning for the convolutional neural network model. Have a look at the Model documentation for an overview of all available hyperparameters. The tuning returns the best model (highest ROC-AUC on the validation data) and an overview of all trained models.

## Methods - Overview

| name | description |
|:-|:-|
| \_\_init\_\_ | Initialize the object with a collection of parameter values. |
| train | Train all models and return the best one. |
## \_\_init\_\_

``` python
def __init__(self, params)
```
Initialize the object with a collection of parameter values. 

 For example: providing {'conv\_num': [1,2,3], 'kernel\_num': [20,50]} will result in training 6 different models (all possible combinations of the provided values) when the train() method is called later on. Parameters that are not provided here will hold their default values in all 6 models. 



| parameter | type | description |
|:-|:-|:-|
| params | dict | A dict containing parameter names as keys and corresponding values as lists. |
## train

``` python
def train(self, data, verbose = True)
```
Train all models and return the best one. 

 Models are evaluated and ranked according to their ROC-AUC on a validation data set. 



| parameter | type | description |
|:-|:-|:-|
| data | pysster.Data | A Data object providing training and validation data sets. |
| verbose | bool | If True, progress information will be printed throughout the training. |

| returns | type | description |
|:-|:-|:-|
| results | tuple(pysster.Model, str) | The best performing model and an overview table of all models are returned. |
