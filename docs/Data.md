# Class Data - Documentation

The Data class provides a convenient way to handle DNA and RNA sequence and structure data for multiple classes. Sequence and structure data are automatically converted into one-hot encoded matrices and split into training/validation/test sets. The data object can then be passed to Grid\_Search or Model objects for easy training and evaluation.

## Methods - Overview

| name | description |
|:-|:-|
| \_\_init\_\_ | Load the sequences and split the data into 70%/15%/15% training/validation/test. |
| train\_val\_test\_split | Randomly split the data into training, validation and test set. |
| get\_labels | Get the labels for a subset of the data. |
| get\_summary | Get an overview of the training/validation/test data for each class. |
## \_\_init\_\_

``` python
def __init__(self, class_files, alphabet)
```
Load the sequences and split the data into 70%/15%/15% training/validation/test. 

 If the goal is to do single-label classification a list of fasta files must be provided (one file per class, the first file will correspond to 'class\_0'). In this case fasta headers are ignored. If the goal is multi-label classification a single fasta file must be provided and headers must indicate class membership as a comma-separated list (e.g. header '\>0,2' means that the sequence belongs to class 0 and 2). 

 For sequence-only files fasta entries have no format restrictions. For sequence-structure files each sequence and structure must span a single line, e.g.: 

  \>0,2  
  CCCCAUAGGGG  
  ((((...)))) (-3.3)  
  SSSSHHHSSSS  
 

 This kind of format is the default output of RNAfold. The third line containing the annotated structure string can be omitted if you want to do the training on the dot-bracket strings (RNAfold will not output the annotated structure string, but we provide a helper function in the utils file to annotate an existing fasta file). **Important: All sequences in all files must have the same length.** 

 The provided alphabet must match the content of the fasta files. For sequence-only files a single string ('ACGT' or 'ACGU') should be provided and for sequence-structure files a tuple should be provided (('ACGU', 'HIMS') to use the annotated structures or ('ACGU', '().') to use dot-bracket structures). 



| parameter | type | description |
|:-|:-|:-|
| class_files | str or [str] | A fasta file (multi-label) or a list of fasta files (single-label). |
| alphabet | str or tuple(str,str) | A string for sequence-only files and a tuple for sequence-structure files. |
## train\_val\_test\_split

``` python
def train_val_test_split(self, portion_train, portion_val, seed = None)
```
Randomly split the data into training, validation and test set. 

 Example: setting portion\_train = 0.6 and portion\_val = 0.3 will set aside 60% of the data for training, 30% for validation and the remaining 10% for testing. Use the seed parameter to get reproducible splits. 



| parameter | type | description |
|:-|:-|:-|
| portion_train | float | Portion of data that should be used for training (<1.0) |
| portion_val | float | Portion of data that should be used for validation (<1.0) |
| seed | int | Seed for the random number generator. |
## get\_labels

``` python
def get_labels(self, group)
```
Get the labels for a subset of the data. 

 The 'group' argument can have the value 'train', 'val', 'test' or 'all'. The returned array has the shape (number of sequences, number of classes). 



| parameter | type | description |
|:-|:-|:-|
| group | str | A string indicating for which subset the labels should be returned. |

| returns | type | description |
|:-|:-|:-|
| labels | numpy.ndarray | An array filled with 0s and 1s indicating class membership. |
## get\_summary

``` python
def get_summary(self)
```
Get an overview of the training/validation/test data for each class. 




| returns | type | description |
|:-|:-|:-|
| summary | str | A tabular overview of every class. |
