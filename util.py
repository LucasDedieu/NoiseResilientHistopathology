import logging
import os
import random
import torch
import argparse
import mlconfig
import datetime



def setup_parsing():
    """
    Parses command-line arguments for configuring the experiment.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Deep feature learning for noisy labels')
    parser.add_argument('--config_path', type=str, default='configs')
    parser.add_argument('--version', type=str, default='baseline')
    parser.add_argument('--exp_name', type=str, default="run1")
    parser.add_argument('--noise_rate', type=float, default=0.0)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resize', type=int, default=224)
    args = parser.parse_args()
    return args



def setup_config(args):
    """
    Loads the configuration file based on the provided arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        tuple: Path to the configuration file and the loaded configuration object.
    """
    config_file = os.path.join(args.config_path, args.version) + '.yaml'
    config = mlconfig.load(config_file)
    return config_file, config



def setup_paths(args, config):
    """
    Sets up necessary directories for the experiment.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        config (mlconfig.Config): Loaded configuration object.

    Returns:
        tuple: Paths for experiment, logging, checkpoints, and results.
    """
    dataset_name = config.dataset.name
    if args.exp_name == '' or args.exp_name is None:
        args.exp_name = 'exp_' + datetime.datetime.now()
    exp_path = os.path.join('experiments', dataset_name, args.version, str(args.noise_rate), args.exp_name)
    log_file_path =  os.path.join(exp_path, 'log')
    checkpoint_path = os.path.join(exp_path, 'checkpoints')
    checkpoint_path_file = os.path.join(checkpoint_path, args.version)
    results_path = os.path.join(exp_path, 'results')
    build_dirs(exp_path)
    build_dirs(log_file_path)
    build_dirs(checkpoint_path)
    build_dirs(results_path)
    return exp_path, log_file_path, checkpoint_path_file, results_path



def setup_logger(args, log_file, level=logging.INFO):
    """
    Sets up the logging mechanism.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        log_file (str): Path to the log file.
        level (int): Logging level.

    Returns:
        logging.Logger: Configured logger.
    """
    formatter = logging.Formatter('%(asctime)s %(message)s')
    #console_handler = logging.StreamHandler()
    #console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(args.version)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    #logger.addHandler(console_handler)
    for arg in vars(args):
        logger.info("%s: %s" % (arg, getattr(args, arg)))
    return logger



def setup_device(args, logger):
    """
    Configures the device for computation (CPU or GPU).

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        logger (logging.Logger): Configured logger.

    Returns:
        torch.device: Configured device.
    """
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        device = torch.device('cuda')
        logger.info("Using CUDA")
        device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
        logger.info("GPU List: %s" % (device_list))
    else:
        device = torch.device('cpu')
        logger.info("No GPU available")

    logger.info("PyTorch Version: %s" % (torch.__version__))
    return device



def build_dirs(path):
    """
    Creates directories if they do not exist.

    Args:
        path (str): Path to the directory.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return