import torch.nn as nn



ModelConfiguration = namedtuple('ModelConfiguration', 
        ['component_latent_dim',
         'background_latent_dim',
         ''])


class EncoderNet(nn.Model):
    """
    General parameterized encoding architecture for VAE components
    """
    def __init__(self, conf):
        super().__init__()

        self.latent_dim = conf.component_latent_dim

        self.network = nn.Sequential()
        self.mlp = nn.Sequqential()
        self.mean_mlp = nn.Sequential()
        self.sigma_mlp = nn.Sequential()

    def forward(self, x):
        x = self.network(x)
        x = self.mlp(x)
        mean = self.mean_mlp(x)
        sigma = self.sigma_mlp(x)
        return mean, sigma


class DecoderNet(nn.Model):
    """
    General parameterized encoding architecture for VAE components
    """
    def __init__(self, conf):
        super().__init__()
        
        self.latent_dim = conf.componetn_latent_dim

        self.network = nn.Sequential()


    def forward(self, x):
        pass


class MaskNet(nn.Model):
    """
    Attention network for mask prediction
    """
    def __init__(self, conf):
        super().__init__()
        
        self.network_structure = nn.Sequential()

    def forward(self, x):
        pass


class SpatialAutoEncoder(nn.Model):
    """
    Spatial transformation and reconstruction auto encoder
    """
    def __init__(self, conf):
        super().__init__()
        self.encoding_network = EncoderNet(conf)
        self.decoding_network = DecoderNet(conf)
        self.mask_network = MaskNet(conf)

    def forward(self, x):
        pass


class MaskedAIR(nn.Model):
    """
    Full model for image reconstruction and mask prediction
    """
    def __init__(self, conf):
        super().__init__()
        self.spatial_vae = SpatialAutoEncoder(conf)
    
    def forward(self, x):
        pass
