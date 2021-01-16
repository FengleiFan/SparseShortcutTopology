<!-- ABOUT THE PROJECT -->
# SparseShortcutTopology

This repository shows you the code regarding the paper [On  a  Sparse  Shortcut  Topology  of  Artificial  Neural  Networks](https://arxiv.org/abs/1811.09003), whose main contribution is to present a promising sparse shortcut topology for deep learning. Besides the theoretical analyses, we conduct comprehensive experiments including prediction and classification experiments to show the superiority of the proposed topology. 


### Struture

* Expressibility experiments
* Generalizability experiments
* Interpretability experiments


### Code Illustration

* Expressibility experiment

We compare the proposed topology to the residual topology in the [neural tangent kernel domain](https://arxiv.org/abs/1904.11955), where the comparisons are mainly dependent on the topology structure and depth, and less influenced by other hyper-parameters. The utilized structure is shown in Figure 1. The prediction result is Figure 2. The main code is "topology_NTK_comparison.ipynb", you may want to open it through Google Colab, through which you do not need to configure the environment for JAX. 

<p align="center">
  <img width="800" src="https://github.com/FengleiFan/SparseShortcutTopology/blob/main/expressibility%20experiment/structure.png" alt="Material Bread logo">
</p>

<p align="center">
  Figure 1. The structure of the proposed model and ResNet utilized in the comparison
</p>

<p align="center">
  <img width="800" src="https://github.com/FengleiFan/SparseShortcutTopology/blob/main/expressibility%20experiment/NTK_results.png" alt="Material Bread logo">
</p>

<p align="center">
  Figure 2. The comparative results
</p>

* Generalizability experiment

**CIFAR100**

**TinyImageNet**

**ImageNet**



<!-- CONTACT -->
## Contact

email: hitfanfenglei@gmail.com



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* []()
* []()
* []()




