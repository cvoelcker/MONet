import collections
from recordtype import recordtype
import argparse

parser = argparse.ArgumentParser(description='Generate a runtime configuration for experiment.')

parser.add_argument('--load_params', action='store_true')
parser.add_argument('--load_location', default='../monet_checkpoints/air_model_test.ckpt')
parser.add_argument('--constrain_theta', action='store_true')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--num_slots', type=int, default=8)
parser.add_argument('--step_size', type=float, default=1e-4)
parser.add_argument('--visdom_env', default='clippingandregularized')
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--gamma', type=float, default=1.0)


MaskedAIRModelConfiguration = recordtype(
        'MaskedAIRModelConfiguration', 
        [
            ('component_latent_dim', 32),
            ('background_latent_dim', 1),
            ('latent_prior', 1.0),
            ('patch_shape', (32, 32)),
            ('image_shape', (256, 256)),
            ('bg_sigma', 0.05),
            ('fg_sigma', 0.1),
            ('num_blocks', 3),
            ('channel_base', 32),
            ('batch_size', 8),
            ('num_slots', 8),
            ('beta', 0.1),
            ('gamma', 0.1),
            ('constrain_theta', True),
            ])


RunConfiguration = recordtype(
        'RunConfiguration',
        [
            ('batch_size', 8),
            ('num_epochs', 50),
            ('vis_every', 50),
            ('visdom_env', 'default'),
            ('load_parameters', False),
            ('step_size', 7e-4),
            ('reshape', False),
            ('summarize', False),
            ('parallel', True),
            ('checkpoint_file', '../monet_checkpoints/air_model_gravitar.ckpt'),
            ('data_dir', '../master_thesis_code/src/data/demon_attack/static_gym_no_white'),
            ])


ExperimentConfiguration = recordtype(
        'ExperimentConfiguration',
        [
            'run_config',
            'model_config',
            ])


masked_air_default_conf = MaskedAIRModelConfiguration()
run_default_conf = RunConfiguration()

def parse_args_to_config(args):
    args = parser.parse_args(args)
    print(args)

    model_conf = masked_air_default_conf
    run_conf = run_default_conf

    run_conf.num_epochs = args.epochs
    run_conf.batch_size = args.batch_size
    run_conf.step_size = args.step_size
    run_conf.visdom_env = args.visdom_env
    run_conf.load_parameters = args.load_params
    run_conf.checkpoint_file = args.load_location

    model_conf.batch_size = args.batch_size
    model_conf.num_slots = args.num_slots
    model_conf.constrain_theta = args.constrain_theta
    model_conf.beta = args.beta
    model_conf.gamma = args.gamma

    full_conf = ExperimentConfiguration(run_conf, model_conf)
    return full_conf
