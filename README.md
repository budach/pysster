## pysster: a Sequence-STructure classifiER  [![Build Status](https://travis-ci.org/budach/pysster.svg?branch=master)](https://travis-ci.org/budach/pysster) [![Build status](https://ci.appveyor.com/api/projects/status/b7kkrb0qu5fsanbh/branch/master?svg=true)](https://ci.appveyor.com/project/budach/pysster/branch/master) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
### Learning Sequence and Structure Motifs in DNA and RNA Sequences using Convolutional Neural Networks

pysster is a python package for training and interpretation of convolutional neural networks. The package can be applied to both DNA and RNA to classify sets of sequences by learning sequence and secondary structure motifs. It can handle multi-class and single-label or multi-label classifications, it offers an automated hyperparameter optimization and options to visualize learned motifs along with information about their positional and class enrichment. The package runs seamlessly on CPU and GPU and provides a simple interface to train and evaluate a network with a handful lines of code.

The preprint can be found on [bioRxiv](https://www.biorxiv.org/content/early/2017/12/06/230086).

If you run into bugs, missing documentation or if you have a feature request, feel free to open an issue.

### Installation

pysster is compatible with Python 3.5+ and can be installed via pip or github.

**Install via pip:**

```sh
pip3 install pysster
```
**Install latest version via github:**
```sh
git clone https://github.com/budach/pysster.git
cd pysster
pip3 install .
```

### Using the GPU

pysster depends on TensorFlow and by default the CPU version of TensorFlow will be installed. If you want to use your NVIDIA GPU (which is recommended for large data sets or grid searchs) make sure that your CUDA and cuDNN drivers are correctly installed and then install the GPU version of TensorFlow:

```sh
pip3 uninstall tensorflow
pip3 install tensorflow-gpu
```

By the time of writing the most recent TensorFlow version is 1.4.0 and requires CUDA 8 and cuDNN 6. You can always check the required versions in the TensorFlow [release notes](https://github.com/tensorflow/tensorflow/releases).


### Documentation

**Tutorials**
* [Example workflow](https://github.com/budach/pysster/blob/master/tutorials/workflow_rna_editing.ipynb) (data loading, model training via grid search, model evaluation + motif visualization showcased using an RNA editing data set)
* [Visualization by optimization of all network layers](https://github.com/budach/pysster/blob/master/tutorials/visualize_all_the_things.ipynb) (an alternative visualization method showcased using an artifical data set)

**API documentation**
* [Data objects](https://github.com/budach/pysster/blob/master/docs/Data.md) (handling and encoding of DNA/RNA input data)
* [Model objects](https://github.com/budach/pysster/blob/master/docs/Model.md) (training and interpretation of networks)
* [Grid_Search objects](https://github.com/budach/pysster/blob/master/docs/Grid_Search.md) (hyperparameter tuning)
* [Motif objects](https://github.com/budach/pysster/blob/master/docs/Motif.md) (motif representation of a PWM)
* [utils functions](https://github.com/budach/pysster/blob/master/docs/utils.md) (save/load Data/Model objects, predict/annotate secondary structures, further processing, etc.)
