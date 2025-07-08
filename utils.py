"""
Author: Liam Laidlaw
Purpose: Miscelaneous utils for the federated learning script. Includes functionality for loading data, sending emails, logging, etc. 
Filename: utils.py
"""

import torch
import torch.nn as nn
from torchvision.transoforms import transforms
from flsim.utils.example_utils import (
    DataLoader,
    DataProvider,
)
from flsim.data.data_sharder import SequentialSharder
import os
import csv
import logging
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, SVHN


from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall, 
    MulticlassF1Score,
    MulticlassAUROC,
)


def build_data_provider(local_batch_size: int, examples_per_user: int, dataset:str = 'cifar10', drop_last: bool = False) -> DataProvider:
    """Builds a data provider for federated learning models. 

    Builds a data provider for a specified dataset. 

    Args:
        local_batch_size: The batch size for local models. 
        examples_per_user: Numer of samples to provide each user.
        dataset: The dataset to load. Defaults to 'cifar10'

    Returns:
        A dataprovider object. 
    """

    _dataset = dataset.lower()
    
    dataset_dispatch = {
        'mnist': 
        {
            'dataset': MNIST, 
            'num_classes': 10,
            'sample_shape': (1, 28, 28),
        },
        'cifar10': 
        {
            'dataset': CIFAR10,
            'num_classes': 10,
            'sample_shape': (1, 3, 32, 32),
        },
        'cifar100':
        {
            'dataset': CIFAR100,
            'num_classes': 100,
            'sample_shape': (1, 3, 32, 32),
        },
        'svhn': 
        {
            'dataset': SVHN,
            'num_classes': 10,
            'sample_shape': (1, 3, 32, 32),
        }, 
    }

    DATASET = dataset_dispatch.get(_dataset).get('dataset', CIFAR10) # if .get fails to locate a dataset, it defaults to cifar10

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = DATASET(
        root="./cifar10", train=True, download=True, transform=transform_train
    )
    test_dataset = DATASET(
        root="./cifar10", train=False, download=True, transform=transform_test
    )
    
    sharder = SequentialSharder(examples_per_shard=examples_per_user)
    fl_data_loader = DataLoader(
        train_dataset, test_dataset, test_dataset, sharder, local_batch_size, drop_last
    )
    data_provider = DataProvider(fl_data_loader)
    print(f"Clients in total: {data_provider.num_train_users()}")
    return data_provider


def send_email(subject: str, body: str) -> None:
    """Sends an email notification. 

    Sends an email from and to the address specified in the .env file. Used for training notifications. If the .env file does not exist, create one with the following format:
        EMAIL_USER='<your username>'
        EMAIL_PASS='<your gmail application passoword>'

    Args:
        subject: The subject line of the email.
        body: The textual content of the email.
    """
    load_dotenv()
    email_user = os.getenv("EMAIL_USER")
    email_pass = os.getenv("EMAIL_PASS")
    if not email_user or not email_pass:
        print("Email username and password not found. Create a .env file with the following format: \nEMAIL_USER='<your username>'\nEMAIL_PASS='<your gmail application passoword>'")
        return
    msg = MIMEText(body)
    msg['Subject'] = f"[ResNet Training] {subject}"
    msg['From'] = email_user
    msg['To'] = email_user
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
            smtp_server.login(email_user, email_pass)
            smtp_server.sendmail(email_user, email_user, msg.as_string())
        logging.info(f"Successfully sent notification email to {email_user}")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")


def save_model(model: nn.Module, config: dict, directory: str = "saved_models", fold: int = None) -> str:
    """Saves a pytorch model with the .pth extension.

    Saves a pytorch model with a particular configuration and training fold to the specified directory with a .pth file extension. 
    If the argued directory does not exit, it is created. 

    Args:
        model: The model to save. 
        config: The configuration for the model. 
        directory: The directory to save the model to. Defaults to 'saved_models/' 
        fold: The fold number for the model. 

    Returns:
        The final path to the model including the model's filename as a string. 
    """
    os.makedirs(directory, exist_ok=True)
    fold_str = f"_fold{fold}" if fold is not None else ""
    filename = f"{config['name']}{fold_str}.pth"
    filepath = os.path.join(directory, filename)
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(state_dict, filepath)
    logging.info(f"SUCCESS: Model state_dict saved to {filepath}")
    return filepath


def save_results_to_csv(results: dict, directory: str = "./", filename: str = "training_results.csv") -> None:
    """Save results to a csv file.

    Saves a dictionary full of training results to the specified directory.
    If the argued directory and/or the argued file does not exist they are created. 

    Args:
        results: Dictionary full of model specifications and training metrics. 
        directory: The directory where the csv file is located. 
        filename: The filename to save results to. Defualts to 'training_results.csv'.
    """

    os.makedirs(directory, exist_ok=True)

    path = os.path.join(directory, filename)
    file_exists = os.path.isfile(path)
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = list(results.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)
    logging.info(f"SUCCESS: Results saved to {filename}")


__all__ = [
    'build_data_provider',
    'send_email',
    'save_model',
    'save_results_to_csv',
]
