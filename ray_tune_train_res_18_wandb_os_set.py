import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import torchvision.transforms as transforms
from torchvision import models, datasets
from torch.optim import lr_scheduler
from typing import Dict, Optional, Any
import time
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from tempfile import TemporaryDirectory
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter
from ray.util.joblib import register_ray
import ray
import argparse
from wandb_osh.ray_hooks import TriggerWandbSyncRayHook


def load_data():
    test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    full_dataset = datasets.ImageFolder(os.path.join(data_path,'asl_train'))
    test_dataset = datasets.ImageFolder(os.path.join(data_path,'asl_test'),test_transform)
    train_dataset, val_dataset = random_split(full_dataset, [0.95, 0.05])

    train_dataset.dataset.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    val_dataset.dataset.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader =  torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    
    class_names = full_dataset.classes

    return train_loader, val_loader, test_loader,{'train': len(train_dataset), 'val': len(val_dataset)}, class_names

def train_model(config):
    print('model loaded')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = models.resnet18(weights='IMAGENET1K_V1')
    train_loader, val_loader, _, dataset_sizes, class_names = load_data()
    for param in model_ft.parameters():
        param.requires_grad = False
    num_ftrs = model_ft.fc.in_features
    if config['normalization']:
        model_ft.fc = nn.Sequential(
            nn.BatchNorm1d(num_ftrs),  
            nn.Linear(num_ftrs, len(class_names))  
        )
    else:
        model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model_ft = nn.DataParallel(model_ft)

    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    
    if config['optimizer'] == 'SGD':
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=config['lr'], momentum=config['momentum'])
    elif config['optimizer'] == 'ADAM':
        optimizer_ft = optim.Adam(model_ft.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']), weight_decay=config['decay'])
    else:
        optimizer_ft = optim.RMSprop(model_ft.parameters(), lr=config['lr'], alpha=config['alpha'], momentum=config['momentum'])
    
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=config['step_size'], gamma=config['gamma'])
    since = time.time()
    dataloaders = {'train' : train_loader, 'val': val_loader}


    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state, scheduler_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            model_ft.load_state_dict(model_state)
            optimizer_ft.load_state_dict(optimizer_state)
            exp_lr_scheduler.load_state_dict(scheduler_state)

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        temp_checkpoint_dir = os.path.join(tempdir, 'checkpoint.pt')

        for epoch in range(config['epochs']):
            print(f"epoch: {epoch}")
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model_ft.train()  # Set model to training mode
                else:
                    model_ft.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                iterator = 0
                for inputs, labels in dataloaders[phase]:
                    if iterator % 100 == 0:
                        print(f'Batch {iterator}/{len(dataloaders[phase])}')
                    iterator += 1
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer_ft.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model_ft(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer_ft.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                if phase == 'train':
                    exp_lr_scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                if phase == 'val':
                    torch.save((model_ft.state_dict(),optimizer_ft.state_dict(), exp_lr_scheduler.state_dict()), temp_checkpoint_dir)    
                    checkpoint = Checkpoint.from_directory(tempdir)
                    print(f"Val Loss: {epoch_loss}, Val Acc: {epoch_acc}")
                    train.report(
                    {"loss": float(epoch_loss), "accuracy": float(epoch_acc)},
                    checkpoint=checkpoint)

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

def test_best_model(best_result):
    config = best_result.config
    _, _, testloader, dataset_sizes, class_names = load_data()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_trained_model = models.resnet18(weights='IMAGENET1K_V1')
    for param in best_trained_model.parameters():
        param.requires_grad = False
    num_ftrs = best_trained_model.fc.in_features
    if config['normalization']:
        best_trained_model.fc = nn.Sequential(
            nn.BatchNorm1d(num_ftrs),  
            nn.Linear(num_ftrs, len(class_names))  
        )
    else:
        best_trained_model.fc = nn.Linear(num_ftrs, len(class_names))
    
    best_trained_model = best_trained_model.to(device)
    criterion = nn.CrossEntropyLoss()
    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")

    model_state, optimizer_state, scheduler_state = torch.load(checkpoint_path)
    torch.save((model_state, optimizer_state, scheduler_state), os.path.join(state_path,'model.pt'))    
    best_trained_model.load_state_dict(model_state)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = best_trained_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    print("Best trial test set accuracy: {}".format(correct / total))

def define_by_run_func(trial) -> Optional[Dict[str, Any]]:
    optimizer = trial.suggest_categorical("optimizer", ["SGD", "ADAM",'RMSprop'])

    trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    if optimizer == "SGD":
        trial.suggest_float("momentum", 0, 0.9)
    elif optimizer == 'ADAM':
        trial.suggest_float("beta1", .9, 0.999)
        trial.suggest_float("beta2", .9, 0.999)
        trial.suggest_float("decay", 0, .2)
    else:
        trial.suggest_float("momentum", 0, 0.9)
        trial.suggest_float('alpha',0.9,0.999)

    trial.suggest_categorical("normalization", [False, True])
    trial.suggest_float('gamma',1e-3,1e-1)
    trial.suggest_int("step_size", 5, 15, step=2)
        
    return {"epochs": max_num_epochs}

def main(num_samples=10, gpus_per_trial=2,cpu_per_trial=1,data_path: str = '.'):
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
        metric='accuracy',
        mode='min')

    algo = OptunaSearch(space=define_by_run_func,metric="accuracy", mode="min")
    algo = ConcurrencyLimiter(algo, max_concurrent=2)
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_model),
            resources={"cpu": cpu_per_trial, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            search_alg=algo,
            num_samples=num_samples
        ),
        run_config=train.RunConfig(
            callbacks=[WandbLoggerCallback(project="Sign language"),TriggerWandbSyncRayHook()]
        )
    )
    results = tuner.fit()
    
    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))

    test_best_model(best_result)

if __name__ == '__main__':
    os.environ["WANDB_MODE"] = "offline"
    if "redis_password" in os.environ:
        print("ip head: ", os.environ["ip_head"])
        print("redis pwd: ", os.environ["redis_password"])
        _node_ip_addr = os.environ["ip_head"].split(":")[0]
        print("node ip addr: ", _node_ip_addr)
        ray.init(
            address=os.environ["ip_head"],
            _redis_password=os.environ["redis_password"],
            _node_ip_address=_node_ip_addr,
        )
        register_ray()
    parser = argparse.ArgumentParser(description="Ray tune training example on slurm") 
    parser.add_argument("--num_samples",required=True,type=int,help="The number of samples.")
    parser.add_argument("--max_num_epochs", required=True,type=int,help="The maximum number of epochs.")
    parser.add_argument( "--gpus_per_trial",required=True,type=int,help="The number of GPUs per trial.")
    parser.add_argument( "--cpus_per_trial",required=True,type=int,help="The number of CPUs per trial.")
    parser.add_argument( "--data_path",required=True,type=str,help="Train and test data path.")
    parser.add_argument( "--state_path",required=True,type=str,help="Path to save model, optimizer and scheduler state")
    args = parser.parse_args()
    global data_path
    global max_num_epochs
    global state_path
    data_path = args.data_path
    max_num_epochs = args.max_num_epochs
    state_path = args.state_path
    main(num_samples=args.num_samples, gpus_per_trial=args.gpus_per_trial, cpu_per_trial=args.cpus_per_trial)