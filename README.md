# Neural Processes

Pytorch implementation of [Neural Processes](https://arxiv.org/abs/1807.01622). This repo follows the
best practices defined in [Empirical Evaluation of Neural Process Objectives](http://bayesiandeeplearning.org/2018/papers/92.pdf).

## Examples

#### Function regression

<img src="https://github.com/EmilienDupont/neural-processes/raw/master/imgs/np_1d.gif" width="400">

#### Image inpainting

<img src="https://github.com/EmilienDupont/neural-processes/raw/master/imgs/celeba.gif" width="256">

## Usage

Simple example of training a neural process on functions or images.

```python
import torch
from neural_process import NeuralProcess, NeuralProcessImg
from training import NeuralProcessTrainer

# Define neural process for functions...
neuralprocess = NeuralProcess(x_dim=1, y_dim=1, r_dim=10, z_dim=10, h_dim=10)

# ...or for images
neuralprocess = NeuralProcessImg(img_size=(3, 32, 32), r_dim=128, z_dim=128,
                                 h_dim=128)

# Define optimizer and trainer
optimizer = torch.optim.Adam(neuralprocess.parameters(), lr=3e-4)
np_trainer = NeuralProcessTrainer(device, neuralprocess, optimizer,
                                  num_context_range=(3, 20),
                                  num_extra_target_range=(5, 10))

# Train on your data
np_trainer.train(data_loader, epochs=30)
```

#### 1D functions

For a detailed tutorial on training and using neural processes on 1d functions, see
the notebook `example-1d.ipynb`.

### Images

To train an image model, use `python main_experiment.py config.json`. This will log information about training and save model weights.

For a detailed tutorial on how to load a trained model and how to use neural processes for inpainting, see the notebook `example-img`. Trained models for MNIST and CelebA are also provided in the `trained_models` folder.

Note, to train on CelebA you will have to download the data from [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

## Acknowledgements

Several people at [OxCSML](https://twitter.com/oxcsml) helped me understand various aspects of neural processes, especially [Kaspar Martens](http://csml.stats.ox.ac.uk/people/martens/), Jef Ton and [Hyunjik Kim](http://csml.stats.ox.ac.uk/people/kim/).

Useful resources:
* Kaspar's [blog post](https://kasparmartens.rbind.io/post/np/)
* Official TensorFlow [implementation](https://github.com/deepmind/neural-processes)

## License

MIT
