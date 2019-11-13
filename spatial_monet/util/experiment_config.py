import collections
from recordtype import recordtype
import argparse

parser = argparse.ArgumentParser(
    description='Generate a runtime configuration for experiment.')

parser.add_argument('--load_params', action='store_true')
parser.add_argument('--load_location',
                    default='checkpoints/default')
parser.add_argument('--data_location',
                    default='test_data')
parser.add_argument('--constrain_theta', action='store_true')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--num_slots', type=int, default=8)
parser.add_argument('--step_size', type=float, default=1e-4)
parser.add_argument('--visdom_env', default='clippingandregularized')
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--gamma', type=float, default=1.0)
parser.add_argument('--latent-dim', type=int, default=16)


MaskedAIRModelConfiguration = recordtype(
    'MaskedAIRModelConfiguration',
    [
        ('latent_dim', 64),
        ('background_latent_dim', 1),
        ('latent_prior', 1.0),
        ('patch_shape', (32, 32)),
        ('image_shape', (64, 64)),
        ('bg_sigma', 0.01),
        ('fg_sigma', 0.05),
        ('num_blocks', 2),
        ('channel_base', 8),
        ('num_slots', 8),
        ('beta', 1.0),
        ('gamma', 1.0),
        ('constrain_theta', True),
    ])

RunConfiguration = recordtype(
    'RunConfiguration',
    [
        ('batch_size', 256),
        ('num_epochs', 50),
        ('vis_every', 50),
        ('visdom_env', 'default'),
        ('load_parameters', False),
        ('step_size', 7e-4),
        ('reshape', False),
        ('summarize', False),
        ('parallel', True),
        ('checkpoint_file', '../monet_checkpoints/air_model_gravitar.ckpt'),
        ('data_dir',
         '../master_thesis_code/src/data/demon_attack/static_gym_no_white'),
    ])

ExperimentConfiguration = recordtype(
    'ExperimentConfiguration',
    [
        'run_config',
        'model_config',
    ])


def record_type_to_dict(recordtype_instance):
    """
    Really dirty hack
    """
    fields = [a for a in recordtype_instance._fields]
    ret_dict = {}
    for field in fields:
        ret_dict[field] = getattr(recordtype_instance, field)
    return ret_dict

masked_air_default_conf = MaskedAIRModelConfiguration()
run_default_conf = RunConfiguration()


def parse_args_to_config(args):
    args = parser.parse_args(args)

    model_conf = masked_air_default_conf
    run_conf = run_default_conf

    run_conf.num_epochs = args.epochs
    run_conf.batch_size = args.batch_size
    run_conf.step_size = args.step_size
    run_conf.visdom_env = args.visdom_env
    run_conf.load_parameters = args.load_params
    run_conf.checkpoint_file = args.load_location
    run_conf.data_dir = args.data_location

    model_conf.num_slots = args.num_slots
    model_conf.constrain_theta = args.constrain_theta
    model_conf.beta = args.beta
    model_conf.gamma = args.gamma
    model_conf.latent_dim = args.latent_dim

    full_conf = ExperimentConfiguration(run_conf, model_conf)
    return full_conf
