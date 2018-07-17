## pysster: a Sequence-STructure classifiER  [![Build Status](https://travis-ci.org/budach/pysster.svg?branch=master)](https://travis-ci.org/budach/pysster) [![Build status](https://ci.appveyor.com/api/projects/status/b7kkrb0qu5fsanbh/branch/master?svg=true)](https://ci.appveyor.com/project/budach/pysster/branch/master) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
### Learning Sequence And Structure Motifs In Biological Sequences Using Convolutional Neural Networks

pysster is a Python package for training and interpretation of convolutional neural networks on biological sequence data. Sequences are classified by learning sequence (and optionally structure) motifs and the package offers sensible default parameters, a hyper-parameter optimization procedure and options to visualize learned motifs. The main features of the package are:

* multi-class and single-label or multi-label classifications
* hyper-parameter tuning (grid search)
* interpretation of learned motifs in terms of positional and class enrichment and motif co-occurrence
* support of input strings over user-defined alphabets (e.g. applicable to DNA, RNA, protein data)
* optional use of structure information, handcrafted features and recurrent layers
* seamless CPU or GPU computation

If you found our tool useful for your work, please cite the accompanying *Bioinformatics* paper ([link](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/bty222/4962494)).
If you run into bugs, missing documentation or if you have a feature request, feel free to open an issue.

### Installation

pysster is compatible with Python 3.5+ and can be installed from PyPI or GitHub.

**Install latest version from GitHub (recommended):**
```sh
git clone https://github.com/budach/pysster.git
cd pysster
pip3 install .
```
**Install from PyPI:**

```sh
pip3 install pysster
```

### Using the GPU

pysster depends on TensorFlow and by default the CPU version of TensorFlow will be installed. If you want to use your NVIDIA GPU (which is recommended for large data sets or grid searchs) make sure that your CUDA and cuDNN drivers are correctly installed and then install the GPU version of TensorFlow:

```sh
pip3 uninstall tensorflow
pip3 install tensorflow-gpu
```

By the time of writing the most recent TensorFlow version is 1.9 and requires CUDA 9 and cuDNN 7. You can always check the required versions in the TensorFlow [release notes](https://github.com/tensorflow/tensorflow/releases).


### Documentation

**Tutorials**
* [Example workflow](https://github.com/budach/pysster/blob/master/tutorials/workflow_rna_editing.ipynb) (data loading, model training via grid search, model evaluation + motif visualization showcased using an RNA editing data set)
* [Visualization by optimization of all network layers](https://github.com/budach/pysster/blob/master/tutorials/visualize_all_the_things.ipynb) (an alternative visualization method showcased using an artifical data set)
* [Limitations of Neural Networks](https://github.com/budach/pysster/blob/master/tutorials/limitations.md) (some critical thoughts on networks applied to sequence data)

**API documentation**
* [Data objects](https://github.com/budach/pysster/blob/master/docs/Data.md) (handling of input data)
* [Model objects](https://github.com/budach/pysster/blob/master/docs/Model.md) (training and interpretation of networks)
* [Grid_Search objects](https://github.com/budach/pysster/blob/master/docs/Grid_Search.md) (hyperparameter tuning)
* [Motif objects](https://github.com/budach/pysster/blob/master/docs/Motif.md) (motif representation of a PWM)
* [utils functions](https://github.com/budach/pysster/blob/master/docs/utils.md) (save/load Data/Model objects, predict/annotate secondary structures, further processing, etc.)


### Changelog

**v1.1.4 - 17. July 2018 (PyPI)**
* added load_additional_positionwise_data() method to Data objects (add arbitrary numerical features for every sequence position; learned features can be visualized for each kernel using the usual Model methods)
* the positive class ("class_0") will now be used as the reference class when computing AUCs in binary classifications (previously the negative class was used)
* some small fixes

**v1.1.3 - 19. March 2018 (PyPI)**
* added visualize_all_kernels() method to Model objects (visualize all kernels at once + get HTML summary report)
* it is now possible to maximize the PR-AUC (precision-recall) instead of the ROC-AUC during a grid search
* changed default color scheme for ACGT and ACGU alphabets to match conventions
* fixed a bug that prevented Data objects from being reproducible
