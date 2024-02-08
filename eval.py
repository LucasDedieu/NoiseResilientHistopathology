import torch
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, average_precision_score
import os
from tqdm import tqdm
import csv
import numpy as np



def eval_model(logger, device, model, test_loader, class_names, results_path, checkpoint_path):    
    """
    Evaluate the performance of a trained model and write results into csv file.

    Args:
        logger (logging.Logger): Logger object for logging information.
        device (torch.device): Device to run the evaluation on.
        model (torch.nn.Module): Trained model to evaluate.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        class_names (list): List of class names.
        results_path (str): Path to save the evaluation results.
        checkpoint_path (str): Path to the checkpoint file containing training information.

    """
    is_binary = (len(class_names)==2)
    total = 0
    correct = 0
    probas = []
    predictions = []
    true_labels = []
    
    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probas.extend(outputs.cpu().numpy())
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        test_acc = correct / total
        test_bacc = balanced_accuracy_score(true_labels, predictions)
        cm = confusion_matrix(true_labels, predictions)
        logger.info(cm)
        checkpoint = torch.load(checkpoint_path + '.pth')
        training_time = checkpoint['training_time']
        val_bacc = checkpoint['val_acc']
        if is_binary:
            prauc = average_precision_score(true_labels, np.array(probas)[:,1])
            logger.info("PR-AUC: %s", prauc)
            column_names = ['test_acc', 'test_bacc', 'val_bacc', 'test_prauc', 'test_confusion_matrix', 'training_time']
            metrics = [test_acc, test_bacc, val_bacc, prauc, cm, training_time]

        else: 
            column_names = ['test_acc', 'test_bacc', 'val_bacc', 'test_confusion_matrix', 'training_time']
            metrics = [test_acc, test_bacc, val_bacc, cm, training_time]

        with open(os.path.join(results_path, "results.csv"), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(column_names)  
            writer.writerow(metrics)
        
        logger.info("Test accuracy: %s",test_acc)
        logger.info("Test balanced accuracy: %s",test_bacc)
        logger.info("Validation balanced accuracy: %s", val_bacc[-1])
        logger.info(classification_report(true_labels, predictions, target_names=class_names, digits=3))
        
        logger.info("Results csv file written at %s", results_path)