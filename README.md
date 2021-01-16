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




![Figure 2](https://github.com/FengleiFan/SparseShortcutTopology/blob/main/expressibility%20experiment/NTK_results.png)
<center>Figure 2. The comparative results.</center>

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/github_username/repo_name.git
   ```
2. Install NPM packages
   ```sh
   npm install
   ```



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_



<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/github_username/repo_name/issues) for a list of proposed features (and known issues).



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email

Project Link: [https://github.com/github_username/repo_name](https://github.com/github_username/repo_name)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* []()
* []()
* []()





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/github_username




Please download and unzip the Tiny-ImageNet dataset from https://github.com/rmccorm4/Tiny-Imagenet-200, and then put my code in the same directory with the data. 
This code is established based on DenseNet implementation. 


