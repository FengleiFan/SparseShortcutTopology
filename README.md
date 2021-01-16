<!-- ABOUT THE PROJECT -->
# SparseShortcutTopology

This repository shows you the code regarding the paper [On  a  Sparse  Shortcut  Topology  of  Artificial  Neural  Networks](https://arxiv.org/abs/1811.09003), whose main contribution is to present a promising sparse shortcut topology for deep learning. Besides the theoretical analyses, we conduct comprehensive experiments including prediction and classification experiments to show the superiority of the proposed topology. 

<p align="center">
  <img width="800" src="https://github.com/FengleiFan/SparseShortcutTopology/blob/main/topology.png" alt="Material Bread logo">
</p>

<p align="center">
  Figure 1. The proposed topology
</p>


### Struture

* Expressibility experiments
* Generalizability experiments
* Interpretability experiments


### Code Illustration

* **Expressibility experiment**

We compare the proposed topology to the residual topology in the [neural tangent kernel domain](https://arxiv.org/abs/1904.11955), where the comparisons are mainly dependent on the topology structure and depth, and less influenced by other hyper-parameters. The utilized structure is shown in Figure 2. The prediction result is Figure 3. The main code is "topology_NTK_comparison.ipynb", you may want to open it through Google Colab, through which you do not need to configure the environment for JAX. 

<p align="center">
  <img width="800" src="https://github.com/FengleiFan/SparseShortcutTopology/blob/main/expressibility%20experiment/structure.png" alt="Material Bread logo">
</p>

<p align="center">
  Figure 2. The structure of the proposed model(a) and ResNet(b) utilized in the comparison
</p>

<p align="center">
  <img width="800" src="https://github.com/FengleiFan/SparseShortcutTopology/blob/main/expressibility%20experiment/NTK_results.png" alt="Material Bread logo">
</p>

<p align="center">
  Figure 3. The comparative results
</p>

* **Generalizability experiment**

**CIFAR100**

How to use?

1. put three files 'optim.py', 's3model.py', 'train_S3Net.py' in the same directory

2. execute: python train_S3Net.py

<p align="center">
  <img width="800" src="https://github.com/FengleiFan/SparseShortcutTopology/blob/main/generalizability%20experiment/cifar100/results.png" alt="Material Bread logo">
</p>

**TinyImageNet**

In this directory, we independently implemented several advanced models on TinyImageNet dataset. All the models are obtained from their official implementation.

How to use?

1. wget http://cs231n.stanford.edu/tiny-imagenet-200.zip and unzip 'tiny-imagenet-200.zip' to get the data.  

2. replace the data directory to your local directory in the "Tiny_xxx.py" (xxx referes to a model being implemented)

3. execute: python Tiny_xxx.py

<p align="center">
  <img width="800" src="https://github.com/FengleiFan/SparseShortcutTopology/blob/main/generalizability%20experiment/TinyImageNet/results.png" alt="Material Bread logo">
</p>


**ImageNet**
In this part, we compare our model with other state-of-the-art models in the small size regime and regular size regime.

How to use?

1. To speed up training, we transform the data into the format of lmdb.

2. for 10-crop validation, execute 'validate_10crop.py'. The obtained model is stored in [Google Drive](https://drive.google.com/drive/folders/1mqJ3nG81vM8Nc_x2sl_dyJY0Gu3yVtw5). 


* **Interpretability experiment**

Shortcuts  can  facilitatetraining by alleviating gradient explosion and vanishing. The mechanism  is  that  shortcuts  provide  additional  paths  thatenable  a  more  accurate  and  easier  gradient  propagation  asthey avoid multiplying gradients many times. We argue thatsuch  a  mechanism  should  also  be  helpful  to  improve  thequality of saliency maps that are based on gradients. What’smore,  in  the  proposed  topology,  shortcuts  directly  connectthe final layer with all previous layers, which is even more
desirable in conveying gradients to the input. Consequently, the  saliency  map  of  the  proposed  topology  should  be  more accurate and sharper.

We apply [FullGrad](https://github.com/idiap/fullgrad-saliency) to compare the saliency maps of the proposed model and other models because FullGrad has desirable properties.

How to use?

1. Download the trained model from [Google Drive](https://drive.google.com/drive/folders/1mqJ3nG81vM8Nc_x2sl_dyJY0Gu3yVtw5).

2. python Interpret_fullgradient.py (you get the saliency maps of models)



3. python Plot_segmentation_map.py (you get the segmentation of saliency maps of models)

<!-- CONTACT -->
## Contact

email: hitfanfenglei@gmail.com



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* []()
* []()
* []()




