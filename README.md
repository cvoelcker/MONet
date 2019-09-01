# MONet
An implementation of the MONet model for unsupervised scene decomposition in PyTorch. It extends the published MONet model by a spatial transformer network, allowing the mask partitions to only cover a part of the picture. This forces the network to adhere more closely to actual object like structures, especially when the objects are not differenciated by color.

## Running the test script

The main.py file contains the setup to run and train the model. It imports a dataset defined by the datasets.py module and works with the experiment config. This config file contains a command line parser for the most important run options:

* --load_params: bool, whether to load the parameters of the network or to run anew
* --load_location: relative file path to save the model parameters
* --constrain_theta: bool, currently deprecated (always true)
* --batch_size: int, batch size for training
* --epochs: int, training epochs
* --num_slots: int, number of maximum masks
* --step_size: float, step size of the Adam optimizer
* --visdom_env: string, name for the visdom and tensorboard logging files
* --beta: float, currently deprecated, disentanglement factor of the variational autoencoder
* --gamma: float, weighing factor for the mask reconstruction loss term


Running the model requires setting the location of the Atari game frames in the settings, or extending the dataset class to deal with other types of images. It expects standard Atari game frames which are not preprocessed in any way.

It is also possible to run the training via the run_with_config.sh shell script. This creates a save file of the current model source and config to save the model during development and not loose progress. This version does not yet handle command line parameters gracefully, so the experiment_config.py file needs to be changed to change the parameters.

The model can also be installed with the setup.py script for easy integration into other projects.
