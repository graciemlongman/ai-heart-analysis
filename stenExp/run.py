import time
import datetime
import albumentations as A
import torch
from utils.utils import *
from utils.metrics import DiceBCELoss
from preparedata import *
from trainer import *

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Vars """
    model_choice='aunet2'
    optim_choice='Adam'
    use_bbox=False
    resume=False

    """ Directories and log file """
    folder = f'{model_choice}_redo/{optim_choice}'
    train_log_path = f"stenExp/model_runs/{folder}/train_log.txt" if not resume else f"stenExp/model_runs/{folder}/train_log_resumed.txt"
    checkpoint_path = f"stenExp/model_runs/{folder}/checkpoint.pth"

    create_dir(f"stenExp/model_runs/{folder}")
    create_file(train_log_path)
    print_and_save(train_log_path, str(datetime.datetime.now()))

    """ Hyperparameters """
    image_size = 256
    size = (image_size, image_size)
    batch_size = 8
    num_epochs = 500
    lr = 1e-4
    early_stopping_patience = 50

    data_str = f"Image Size: {size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    data_str += f"Early Stopping Patience: {early_stopping_patience}\n"
    print_and_save(train_log_path, data_str)

    """ Data augmentation: StenUNet Transforms """
    transform = A.Compose([
        A.Affine(scale=(0.7, 1.4), p=0.5),
        A.Rotate(limit=180, p=0.5),
        A.GaussNoise(std_range=(0.0, 0.32), p=0.5),
        A.GaussianBlur(blur_limit=3, sigma_limit=(0.71, 1), p=0.5)
        ])
    
    """ Dataset """
    # print('Preprocessing dataset...')
    # prepare_data_stenosis(copy_data=True)

    train_loader, valid_loader = data_loader(train_log_path, use_bbox, size, transform, batch_size)

    #check_labels(train_loader, valid_loader, zip(test_x,test_y))

    """ Model """
    device = torch.device('cuda')
    model = ModelZoo(choice=model_choice, partition='train').to(device)

    if resume:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    optimizer = OptZoo(choice=optim_choice, model=model, lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    loss_fn = DiceBCELoss()
    loss_name = "BCE Dice Loss"
    
    data_str = f"Model: {model_choice}\nOptimizer: {optim_choice}\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    """ Training the model """
    print('Beginning training...')
    best_valid_metrics = 0.0
    early_stopping_count = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        torch.autograd.set_detect_anomaly(True)
        train_loss, train_metrics = train(model, train_loader, optimizer, loss_fn, device, use_bbox)
        valid_loss, valid_metrics = evaluate(model, valid_loader, loss_fn, device, use_bbox)
        scheduler.step(valid_loss)
 
        if valid_metrics[1] > best_valid_metrics:
            data_str = f"Valid F1 improved from {best_valid_metrics:2.4f} to {valid_metrics[1]:2.4f}. Saving checkpoint: {checkpoint_path}"
            print_and_save(train_log_path, data_str)

            best_valid_metrics = valid_metrics[1]
            torch.save(model.state_dict(), checkpoint_path)
            early_stopping_count = 0

        if valid_metrics[1] < best_valid_metrics:
            early_stopping_count += 1

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        data_str += f"\tTrain Loss: {train_loss:.4f} - Jaccard: {train_metrics[0]:.4f} - F1: {train_metrics[1]:.4f} - Recall: {train_metrics[2]:.4f} - Precision: {train_metrics[3]:.4f}\n"
        data_str += f"\t Val. Loss: {valid_loss:.4f} - Jaccard: {valid_metrics[0]:.4f} - F1: {valid_metrics[1]:.4f} - Recall: {valid_metrics[2]:.4f} - Precision: {valid_metrics[3]:.4f}\n"
        print_and_save(train_log_path, data_str)

        if early_stopping_count == early_stopping_patience:
            data_str = f"Early stopping: validation loss stops improving from last {early_stopping_patience} continously.\n"
            print_and_save(train_log_path, data_str)
            break

    """ Plot training graph """
    if resume:
        train_log_path=[f"stenExp/model_runs/{folder}/train_log.txt", f"stenExp/model_runs/{folder}/train_log_resumed.txt"]

    plot_training_curve(train_log_path, f"stenExp/model_runs/{folder}/loss_epoch_curve.png")

    print('fin')