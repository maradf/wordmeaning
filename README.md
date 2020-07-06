# Understanding Interpersonal Relationships with an LSTM

This repository contains the code and results of an exploratory research done for the course Cognitive and Computational Aspects of Word Meaning at Utrecht University. The goal of this research was find the point at what number of individuals an LSTM, either unidirectional or bidirectional, is no longer able to understand the interpersonal relationships of these individuals. These relationships are the ones of A's parent B, B's child A, A's friend C or C's enemy B, with A, B and C being individuals. ```train.py``` trains the model, when given the correct parameters, and will save the best model to the folder, along with files containing the accuracies. The functions found in ```plots.py``` plot these accuracies, after specifying where the files are saved. The accompanying paper of this project can be found in the file ```Paper Understanding Interpersonal Relationships with an LSTM.pdf```

## Getting Started

To get this code running, simply clone this repository to your machine, and install all necessary prerequisites described below. 

### Prerequisites

The software needed to run this project is Python 3.7 (https://www.python.org/downloads/), and the package Pytorch (https://pytorch.org/get-started/locally/). The other packages, which can be installed using ```pip install <package name>``` are
-  ```sys```
-  ```os```
-  ```shutil```
-  ```random```
-  ```numpy```
-  ```pandas```
-  ```matplotlib```
-  ```pickle```

### Running the code

To train a model, ```train.py``` needs to be run. This can be done as follows

```
$ python3 train.py 2 4 3 l 
```
In this example, the model is trained with 2 pairs of individuals (so four individuals in total), has four relationships (parent, child, friend, enemy), it is trained on a maximum complexity of three (thus the most intricate relationship described would describe three individuals, such as the phrase A's friend's parent's C), and the phrases are left-branching (A's friend B as opposed to B friend of A). 

There are also optional arguments that can be given, such as the following:
```
$ python3 train.py 2 4 3 l 0.85 bidirectional
```
This way, the percentage of sentences of the maximum complexity (in this case 3) used in the trainingset is 0.85 (standard is 0.8 for maximum results), and the model is trained bidirectionally. If ```bidirectional``` is not in one of the arguments, the model is automaticaly unidirectional.

After training a model, a plot can be created by using one of the functions in ```plots.py```.
