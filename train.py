import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
import gc
from tqdm import tqdm
import time
    

def train_model(config, logger, device, model, train_loader, val_loader, optimizer, scheduler, start_epoch=0, epochs_before_stop=-1, best_v_loss=float('inf'), v_acc=[], training_time=0, checkpoint_path=""):
    """
    Train a given model using the specified parameters and data loaders.

    Args:
        config (object): Configuration object containing training parameters.
        logger (logging.Logger): Logger object for logging information.
        device (torch.device): Device to run the training on (e.g., CPU or GPU).
        model (nn.Module): The neural network model to be trained.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        optimizer (torch.optim): Optimizer.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        start_epoch (int, optional): Starting epoch number. Defaults to 0.
        epochs_before_stop (int, optional): Number of epochs before early stopping. Defaults to -1.
        best_v_loss (float, optional): Best validation loss achieved so far. Defaults to float('inf').
        v_acc (list, optional): List of validation accuracies. Defaults to [].
        training_time (int, optional): Total training time. Defaults to 0.
        checkpoint_path (str, optional): Path to save model checkpoints. Defaults to "".

    Returns:
        tuple: A tuple containing lists of training loss, validation loss, validation accuracy, precision, recall, and F1 score.
    """
    criterion = config.criterion()
    n_epoch = config.epochs
    patience = config.patience

    warmup = config.warmup.active
    logger.info("Warmup: %s", warmup)
    if warmup:
        warmup_ep = config.warmup.epochs
        warmup_lr = config.warmup.lr
        warmup_loss = nn.CrossEntropyLoss()
        if int(start_epoch) > warmup_ep:
            warmup = False

    t_loss = []
    v_loss = []
    v_acc = v_acc
    v_f1 = []
    v_precision = []
    v_recall = []

    if epochs_before_stop == -1:
        epochs_before_stop = patience

    training_lr = scheduler.get_last_lr()[0]
    loss_func = criterion

    for epoch in range(start_epoch, n_epoch):
        start_time = time.time()
        running_loss = 0.0
        val_loss = 0.0
        correct = 0
        total = 0
        
        #WARMUP
        if warmup:
            if epoch < warmup_ep:
                loss_func = warmup_loss
                for g in optimizer.param_groups:
                    g['lr'] = warmup_lr
            elif epoch == warmup_ep:
                best_v_loss = float('inf')
                loss_func = criterion
                for g in optimizer.param_groups:
                    g['lr'] = training_lr
                warmup = False
        
        #TRAIN
        for data in tqdm(train_loader):
            images, labels = data
            labels = labels.type(torch.LongTensor)
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*images.size(0)
 
        #VALIDATION
        predictions = []
        true_labels = []
        model.eval()
        with torch.no_grad():
            for data in val_loader:
                images, labels = data
                labels = labels.type(torch.LongTensor)
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = loss_func(outputs, labels)
                val_loss += loss.item()*images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = balanced_accuracy_score(true_labels, predictions)
        epoch_train_loss = running_loss/len(train_loader.dataset)
        epoch_val_loss = val_loss/len(val_loader.dataset)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, pos_label=1, average = "macro", zero_division=0.)
        v_precision.append(precision)
        v_recall.append(recall)
        v_f1.append(f1)
        v_acc.append(accuracy)
        t_loss.append(epoch_train_loss)
        v_loss.append(epoch_val_loss)
        logger.info(f'epoch {epoch + 1} -> t_loss: {epoch_train_loss:.4f}, v_loss: {epoch_val_loss:.3f}, bacc: {accuracy:.4f}, precision:{precision:.2f}, recall:{recall:.2f}, f1:{f1:.2f}')

        if scheduler:
            scheduler.step()
        
        #EARLY STOPPING
        if epoch_val_loss < best_v_loss:
            best_v_loss = epoch_val_loss
            epochs_before_stop = patience
        else:
            epochs_before_stop -= 1

        if epochs_before_stop == 0:
            logger.info('Early Stopping : %s epochs since last best val loss. Total epochs trained : %s',patience ,epoch+1)
            return t_loss, v_loss, v_acc, v_precision, v_recall, v_f1

        
        end_time = time.time()
        training_time += (end_time - start_time)
        
        #CHECKPOINT 
        model.save(checkpoint_path, epoch, epochs_before_stop, best_v_loss, optimizer, scheduler, v_acc, training_time)
 

    return t_loss, v_loss, v_acc, v_precision, v_recall, v_f1
