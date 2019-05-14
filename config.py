from collections import namedtuple
import os

config_options = [
    # Training config
    'vis_every',  # Visualize progress every X iterations
    'batch_size',
    'num_epochs',
    'load_parameters',  # Load parameters from checkpoint
    'checkpoint_file',  # File for loading/storing checkpoints
    'data_dir',  # Directory for the training data
    'parallel',  # Train using nn.DataParallel
    # Model config
    'num_slots',  # Number of slots k,
    'num_blocks',  # Number of blochs in attention U-Net 
    'channel_base',  # Number of channels used for the first U-Net conv layer
    'bg_sigma',  # Sigma of the decoder distributions for the first slot
    'fg_sigma',  # Sigma of the decoder distributions for all other slots
    'reshape', # flag to reshape the input picture to a specific size
    'latent_dim', # enables setting the size of the spatial transform glimpse
    'z_dim', # the size of the latent embedding in the VAE component
]

MonetConfig = namedtuple('MonetConfig', config_options)
MonetConfig.__new__.__defaults__ = (None,) * len(MonetConfig._fields)

sprite_config = MonetConfig(vis_every=50,
                            batch_size=64,
                            num_epochs=20,
                            load_parameters=True,
                            checkpoint_file='./checkpoints/sprites.ckpt',
                            data_dir='./data/',
                            parallel=True,
                            num_slots=4,
                            num_blocks=5,
                            channel_base=64,
                            bg_sigma=0.09,
                            fg_sigma=0.11,
                           )

clevr_config = MonetConfig(vis_every=50,
                           batch_size=64,
                           num_epochs=200,
                           load_parameters=True,
                           checkpoint_file='/work/checkpoints/clevr64.ckpt',
                           data_dir=os.path.expanduser('~/data/CLEVR_v1.0/images/train/'),
                           parallel=True,
                           num_slots=11,
                           num_blocks=6,
                           channel_base=64,
                           bg_sigma=0.09,
                           fg_sigma=0.11,
                          )

atari_config = MonetConfig(vis_every=50,
                           batch_size=2,
                           num_epochs=20,
                           load_parameters=False,
                           checkpoint_file='../monet_checkpoints/atari_tiny_net.ckpt',
                           data_dir='../master_thesis_code/src/data/static_gym/',
                           parallel=False,
                           num_slots=15,
                           num_blocks=1,
                           channel_base=32,
                           bg_sigma=0.09,
                           fg_sigma=0.11,
                           reshape=True,
                          )

spatial_transform_config = MonetConfig(vis_every=50,
                           batch_size=2,
                           num_epochs=20,
                           load_parameters=False,
                           checkpoint_file='../monet_checkpoints/atari_tiny_net.ckpt',
                           data_dir='../master_thesis_code/src/data/static_gym/',
                           parallel=False,
                           num_slots=15,
                           num_blocks=3,
                           channel_base=32,
                           bg_sigma=0.09,
                           fg_sigma=0.11,
                           reshape=True,
                           latent_dim=(32,32),
                           z_dim=32,
                          )
