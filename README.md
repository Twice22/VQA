
# Visual Question Answering

## Presentation
This project is part of an assignment of the Object Recognition and Computer Vision course that I've followed  at the [Ens Paris-Saclay](https://en.wikipedia.org/wiki/%C3%89cole_normale_sup%C3%A9rieure_Paris-Saclay) ([master MVA](http://math.ens-paris-saclay.fr/version-francaise/formations/master-mva/contenus-/master-mva-cours-2017-2018-161721.kjsp?RH=1242415112528)). The goal was to implement the models of the following article https://arxiv.org/pdf/1505.00468.pdf and to discuss both quantitavely and qualitavely the results obtain.

## Getting Started
If you want to use this code on your own personal computer you must have a CUDA compatible GPU (nvidia GTX). You must also have enough
RAM on your computer (8 GB would be enough). As I've implemented the code for windows, the following installation instructions are for windows users.

### Installation
1. Install [CUDA 8.0](https://developer.nvidia.com/cuda-80-ga2-download-archive) for your operating system (At the time I'm writing this note, tensorflow is not compatible with higher version on windows environment)
2. Launch the installer and install CUDA
3. Download cuDNN 5.1 for Windows (At the time I'm writing these lines, tensorflow only supports cuDNN 5.1 for windows users)
4. Unzip the archive. You should get a folder containing 3 other folders:
	+ bin
	+ include
	+ lib
5. Go to `C:\` and create a folder named `Cuda`, then copy and paste the folders `bin`, `include`, `lib` inside your `Cuda` folder.
6. Add `C:\Cuda` to your Path environment variable. To do so:
	+ Right click on `Windows -> System -> Advanced system settings (on the left) -> Environment Variables`
	+ Click on `Path Variable` under `System Variables` and then click `Edit...` and Add `;C:\Cuda` at the end of the `Path` variable. (On windows 10 you just have to add `C:\Cuda` on a new line).
7. Download and install [Anaconda with python 3.6 (x64)](https://www.anaconda.com/download/). Once Anaconda is installed, open an anaconda prompt and type:
	```bash
	pip install --ignore-installed --upgrade tensorflow-gpu
	```
8. Install spacy: open an anaconda prompt with __ADMIN RIGHT__ and type:
	```bash
	python -m spacy download en_vectors_web_lg
	```
9. Download and install [Graphviz](https://graphviz.gitlab.io/_pages/Download/Download_windows.html)
then add Graphviz to your environment variable `Path`:
`Computer > Properties > Advanced system settings > Environment Variables`
and add `;C:\Program Files (x86)\Graphviz2.38\bin` (your path can be different)

10. Reinstall pip (known bug that after installing tensorflow, pip is broken...) by typing in an anaconda prompt:
	```bash
	conda install pip
	```
11. Install keras by typing in an anaconda prompt either
	```
	conda install keras
	```
	or
	```
	pip install keras
	```

12. Install pydot-ng (this will allow you to see the architecture of your neural network) by typing in an anaconda prompt:
	```bash
	pip install pydot-ng
	```

### Descriptions

__Folders__
+ Annotations: unzip the content of this [file](http://images.cocodataset.org/annotations/annotations_trainval2014.zip) in Annotations folder (Annotations folder should contains 6 json files)
+ COCO
	+ __annotations__: unzip the content of these [file 1](http://images.cocodataset.org/annotations/image_info_test2014.zip), [file 2](http://images.cocodataset.org/annotations/image_info_test2015.zip), [file 3](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) in this folder. (The folder should contains only json files)
	+ __images__: images folder contains both validation and training images from the COCO dataset. The training images are available [here](http://images.cocodataset.org/zips/train2014.zip) and the validation images are available [here](http://images.cocodataset.org/zips/val2014.zip). (This folder should contains only jpeg images)
	+  __images_test__: images_test folder contains the testing images from the COCO dataset. These images can be downloaded [here](http://images.cocodataset.org/zips/test2014.zip). (This folder should only contains jpeg images)
+ histories:
	+ BOWQ_I: contains training and validation accuracy/loss at each epoch. This file is created when executing `BOWQ_I.py` .
	+  LSTM_Q: contains training and validation accuracy/loss at each epoch. This file is created when executing `LSTM_Q.py` .
	+  LSTMQ_I: contains training and validation accuracy/loss at each epoch. This file is created when executing `LSTMQ_I.py` .
	+ plots.ipynb: jupyter notebook that plots the training and validation accuracy/loss for all models (BOWQ_I, LSTM_Q and LSTMQ_I)
+ models: contains `vgg16.py` a python script that allows to recover the fc7 layer of the VGG16 neural network (the fc7 leayer corresponds to the features computed on the images. These features are vectors of size 4096)
+ our_images: put your on images here if you want to test the model on your own images (Note: the code to test on your on image is already provided for each model in the `online_modelname.ipynb` where modelname is either `LSTM_Q`, `LSTMQ_I`, `BOWQ_I`)
+ preprocess_datas:
	+ the file contains in this folder are created by the scripts: `create_dict.ipynb`,`features_extractor.ipynb` and  `create_all_answers.ipynb` (Note: you should execute these scripts to recreate the files of this directory as they are to heavy to let me upload them on github)
+ Questions: this folder contains json files that encode the questions and answers. These files can be downloaded from [here](http://www.visualqa.org/vqa_v1_download.html) (Note: you can also download the v2 version available [here](http://www.visualqa.org/download.html), but in the experiment I've focus on the v1 version)
+ weights: this folder contains the following directories:
	+ BOWQ_I: contains the weights of the neural-network saved at each epoch. These weights are generated while executing the python script: `BOWQ_I.py`
	+ LSTM_Q: contains the weights of the neural-network saved at each epoch. These weights are generated while executing the python script: `LSTM_Q.py`
	+ LSTMQ_I: contains the weights of the neural-network saved at each epoch. These weights are generated while executing the python script: `LSTMQ_I.py`

__Scripts__
+ baselines.ipynb: this script implements the baselines of the article (random, qPrior, per q-Prior, nearest-neigbhors)
+ BOWQ_I.py: this script train the Bag-of-Word + Image feature model. Just type `python BOWQ_I.py` in your shell to execute the code (Note: you can change the model, the number of epochs and so on by editing the python file)
+ LSTM_Q.py: this script train the Bag-of-Word + Image feature model. Just type `python LSTM_Q.py` in your shell to execute the code (Note: you can change the model, the number of epochs and so on by editing the python file)
+ LSTMQ_I.py: this script train the Bag-of-Word + Image feature model. Just type `python LSTMQ_I.py` in your shell to execute the code (Note: you can change the model, the number of epochs and so on by editing the python file)
+ `create_all_answers.ipynb`, `create_dict.ipynb`, `features_extractor.ipynb` are scripts that you should execute before any other scripts as it will preprocess the data and create files.
+ `features_processor.py` and `utils.py` contains functions that are used by other scripts. All the functions are documented by docstrings so you can understand what is the purpose of each functions.
+ `online_modelname.ipynb` are jupyter notebook that allows you to test the model on either the testing images or your own images. These file also provide the final accuracy of each model computed on the validation set.
+ `cb.py`: this script is used in `LSTM_Q.py`, `LSTM_Q.py` and `LSTMQ_I.py` to save the training and validation accuracy/loss at each epochs. It is a callback.

### Notes
The code can be improved in many ways. I've made the choice to use function rather than files to preprocess the data as it is easier if we want to change the settings of the neural network. For example to use K=2000 top answers instead of K=1000 top answers (see the [paper](https://arxiv.org/pdf/1505.00468.pdf) to understand the meaning), one can just use
```python
	topKFrequentAnswer(data_q, data_a, data_qval, data_aval, K=2000)
```

with K=2000 to preprocess the data.
