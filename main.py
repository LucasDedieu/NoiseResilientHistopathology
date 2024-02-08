import torch
import argparse
import util
import os
import datetime
import random
import mlconfig
import losses
import models
import datasets
import shutil
import custom_transforms
from torch.utils.data import DataLoader
from train import train_model
from eval import eval_model
import sys
mlconfig.register(torch.optim.SGD)
mlconfig.register(torch.optim.lr_scheduler.CosineAnnealingLR)



# ArgParse
parser = argparse.ArgumentParser(description='Deep feature learning for noisy labels')
parser.add_argument('--config_path', type=str, default='configs')
parser.add_argument('--version', type=str, default='baseline')
parser.add_argument('--exp_name', type=str, default="run1")
parser.add_argument('--noise_rate', type=float, default=0.0)
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--resize', type=int, default=224)
args = parser.parse_args()

# Set up
config_file = os.path.join(args.config_path, args.version) + '.yaml'
config = mlconfig.load(config_file)
config.set_immutable()
dataset_name = config.dataset.name

if args.exp_name == '' or args.exp_name is None:
    args.exp_name = 'exp_' + datetime.datetime.now()
exp_path = os.path.join('experiments', dataset_name, args.version, str(args.noise_rate), args.exp_name)
log_file_path =  os.path.join(exp_path, 'log')
checkpoint_path = os.path.join(exp_path, 'checkpoints')
checkpoint_path_file = os.path.join(checkpoint_path, args.version)
results_path = os.path.join(exp_path, 'results')

util.build_dirs(exp_path)
util.build_dirs(log_file_path)
util.build_dirs(checkpoint_path)
util.build_dirs(results_path)

logger = util.setup_logger(name=args.version, log_file=log_file_path + ".log")
for arg in vars(args):
    logger.info("%s: %s" % (arg, getattr(args, arg)))

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

shutil.copyfile(config_file, os.path.join(exp_path, args.version+'.yaml'))
for key in config:
    logger.info("%s: %s" % (key, config[key]))



def main():
    class_names, X_train, y_train, X_val, y_val, X_test, y_test = datasets.get_dataset(device, config, logger, noise_rate=args.noise_rate, seed=args.seed)
    num_classes = len(class_names)
    if config.dataset.type == "images":
        train_set = datasets.ImgDataset(X_train, y_train, transform=custom_transforms.train_images(resize=args.resize))
        val_set = datasets.ImgDataset(X_val, y_val, transform=custom_transforms.val_images(resize=args.resize))
        test_set = datasets.ImgDataset(X_test, y_test, transform=custom_transforms.val_images(resize=args.resize))
        model = models.ResNet50(pretrained=True, num_classes=num_classes)
    elif config.dataset.type == "deep_features":
        input_size = X_train.shape[1]
        print(X_train.shape)
        train_set = datasets.DFDataset(X_train, y_train, transform=custom_transforms.train_df(sigma=config.dataset.sigma))
        val_set = datasets.DFDataset(X_val, y_val)
        test_set = datasets.DFDataset(X_test, y_test)
        model = models.ClassificationHead(input_size=input_size, num_classes=num_classes)
    else:
        logger.error("Unknow dataset type. Available: 'images', 'deep_features'")
        sys.exit(1)
    
    train_loader = DataLoader(train_set, batch_size=config.dataset.batch_size, num_workers=config.dataset.num_workers, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.dataset.batch_size)
    test_loader = DataLoader(test_set, batch_size=config.dataset.batch_size)
    #val_loader=test_loader

    logger.info("Dataset loaded")
    logger.info("train: %s, val: %s, test: %s", len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))        

    model.to(device)

    optimizer=config.optimizer(model.parameters())
    scheduler = config.scheduler(optimizer)

    
    if args.resume:
        checkpoint = model.load_model(path=checkpoint_path_file,
                                        optimizer=optimizer,
                                        scheduler=scheduler)
        start_epoch = checkpoint['epoch'] + 1
        epochs_before_stop = checkpoint['epochs_before_stop']
        best_v_loss = checkpoint['best_val_loss']
        v_acc = checkpoint['val_acc']
        training_time = checkpoint['training_time']
        logger.info("File %s loaded" % (checkpoint_path_file))

    else:
        start_epoch = 0
        epochs_before_stop = -1
        best_v_loss = float('inf')
        v_acc = []
        training_time = 0
        
    if start_epoch < config.epochs:
        logger.info("Starting training...")
        train_model(config=config,
                    logger=logger,
                    device=device, 
                    model=model, 
                    train_loader=train_loader, 
                    val_loader=test_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    start_epoch=start_epoch,
                    epochs_before_stop=epochs_before_stop,
                    best_v_loss=best_v_loss,
                    v_acc=v_acc,
                    training_time=training_time,
                    checkpoint_path=checkpoint_path_file
        )
    logger.info("Model successfully trained")

    logger.info("Starting model evaluation...")
    eval_model(logger, device, model, test_loader, class_names, results_path, checkpoint_path_file)
    sys.exit(0)



if __name__ == '__main__':
    main()