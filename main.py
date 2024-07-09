import torch
import util
import os
import models
import datasets
import shutil
import custom_transforms
from torch.utils.data import DataLoader
from train import train_model
from eval import eval_model
import sys
import mlconfig
# Registering necessary modules with mlconfig
mlconfig.register(torch.optim.SGD)
mlconfig.register(torch.optim.lr_scheduler.CosineAnnealingLR)


def setup():
    args = util.setup_parsing()
    config_file, config = util.setup_config(args)
    exp_path, log_file_path, checkpoint_path_file, results_path = util.setup_paths(args, config)
    logger = util.setup_logger(args, log_file=log_file_path + ".log")
    device = util.setup_device(args, logger)

    shutil.copyfile(config_file, os.path.join(exp_path, args.version+'.yaml'))
    for key in config:
        logger.info("%s: %s" % (key, config[key]))

    return args, config, checkpoint_path_file, results_path, logger, device


def main():
    args, config, checkpoint_path_file, results_path, logger, device = setup()

    class_names, X_train, y_train, X_val, y_val, X_test, y_test = datasets.get_dataset(device, config, logger, noise_rate=args.noise_rate, seed=args.seed)
    num_classes = len(class_names)

    if config.dataset.type == "images":
        train_set = datasets.ImgDataset(X_train, y_train, transform=custom_transforms.train_images(resize=args.resize))
        val_set = datasets.ImgDataset(X_val, y_val, transform=custom_transforms.val_images(resize=args.resize))
        test_set = datasets.ImgDataset(X_test, y_test, transform=custom_transforms.val_images(resize=args.resize))
        model = models.ResNet50(pretrained=True, num_classes=num_classes)

    elif config.dataset.type == "deep_features":
        input_size = X_train.shape[1]
        logger.info("Features dim %s", input_size)
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

    logger.info("Dataset loaded")
    logger.info("train: %s, val: %s, test: %s", len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))        

    model.to(device)

    optimizer = config.optimizer(model.parameters())
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
                    val_loader=val_loader,
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