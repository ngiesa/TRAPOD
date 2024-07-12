# -*- coding: utf-8 -*-
"""
Created on Sat May 12 16:49:49 2018

@author: Zhiyong
"""
import torch.cuda

from losses import weighted_binary_cross_entropy
from GRUD import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torcheval.metrics import BinaryAUROC
from torcheval.metrics.functional import binary_auprc

def Train_Model(model, train_dataloader, valid_dataloader, num_epochs=1, patience=10, min_delta=0.00001,
                learning_rate = 0.0001, batch_size=64):
    print('Model Structure: ', model)
    print('Start Training ... ')

    model = model.to(device)

    if (type(model) == nn.modules.container.Sequential):
        output_last = model[-1].output_last
        print('Output type dermined by the last layer')
    else:
        output_last = model.output_last
        print('Output type dermined by the model')

    
    # configure weighted binary cross entropy loss 

    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99)
    use_gpu = torch.cuda.is_available()

    losses_train = []
    losses_valid = []
    losses_epochs_train = []
    losses_epochs_valid = []

    cur_time = time.time()
    pre_time = time.time()

    # Variables for Early Stopping
    is_best_model = 0
    patient_epoch = 0
    for epoch in range(num_epochs):

        print("start epoch: ", epoch)

        trained_number = 0

        valid_dataloader_iter = iter(valid_dataloader)

        losses_epoch_train = []
        losses_epoch_valid = []

        for data in train_dataloader:
            inputs, labels = data

            if inputs.shape[0] != batch_size:
                continue

            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            model.zero_grad()

            outputs = model(inputs)

            loss_train = weighted_binary_cross_entropy(inputs=torch.squeeze(outputs), targets=torch.squeeze(labels), prevalence=0.09)
            
            print("training loss: ", loss_train)

            losses_train.append(loss_train.data)
            losses_epoch_train.append(loss_train.data)

            optimizer.zero_grad()

            loss_train.backward()

            # perform gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            
            optimizer.step()

            # validation
            try:
                inputs_val, labels_val = next(valid_dataloader_iter)
            except StopIteration:
                valid_dataloader_iter = iter(valid_dataloader)
                inputs_val, labels_val = next(valid_dataloader_iter)

            if use_gpu:
                inputs_val, labels_val = Variable(inputs_val.cuda()), Variable(labels_val.cuda())
            else:
                inputs_val, labels_val = Variable(inputs_val), Variable(labels_val)

            model.zero_grad()

            outputs_val = model(inputs_val)

            
            # correct nan outputs
            outputs_val = torch.nan_to_num(x=outputs_val, nan=np.float64(outputs_val.nanmean()))
            
            loss_valid = weighted_binary_cross_entropy(inputs=torch.squeeze(outputs_val), targets=torch.squeeze(labels_val), prevalence=0.09)
            
            sig_act = nn.Sigmoid()
            out_val_prob = sig_act(outputs_val)
            
            print("validation loss: ", loss_valid)
            
            auroc = BinaryAUROC()
            auroc.update(torch.squeeze(out_val_prob), torch.squeeze(labels_val))
            auroc_val = auroc.compute()
            
            auprc_val = binary_auprc(torch.squeeze(out_val_prob), torch.squeeze(labels_val))
            
            
            losses_valid.append(loss_valid.data)
            losses_epoch_valid.append(loss_valid.data)

            # output
            trained_number += 1

        avg_losses_epoch_train = sum(losses_epoch_train).cpu().numpy() / float(len(losses_epoch_train))
        avg_losses_epoch_valid = sum(losses_epoch_valid).cpu().numpy() / float(len(losses_epoch_valid))
        losses_epochs_train.append(avg_losses_epoch_train)
        losses_epochs_valid.append(avg_losses_epoch_valid)

        # Early Stopping
        if epoch == 0:
            is_best_model = 1
            best_model = model
            min_loss_epoch_valid = 10000.0
            if avg_losses_epoch_valid < min_loss_epoch_valid:
                min_loss_epoch_valid = avg_losses_epoch_valid
        else:
            if min_loss_epoch_valid - avg_losses_epoch_valid > min_delta:
                is_best_model = 1
                best_model = model
                min_loss_epoch_valid = avg_losses_epoch_valid
                patient_epoch = 0
            else:
                is_best_model = 0
                patient_epoch += 1
                if patient_epoch >= patience:
                    print('Early Stopped at Epoch:', epoch)
                    break

        # Print training parameters
        cur_time = time.time()
        print('Epoch: {}, train_loss: {}, valid_loss: {}, time: {}, best model: {}'.format(
            epoch,
            np.around(avg_losses_epoch_train, decimals=8),
            np.around(avg_losses_epoch_valid, decimals=8),
            np.around([cur_time - pre_time], decimals=2),
            is_best_model))
        pre_time = cur_time

    return best_model, [losses_train, losses_valid, losses_epochs_train, losses_epochs_valid]


def Test_Model(model, test_dataloader):
    if (type(model) == nn.modules.container.Sequential):
        output_last = model[-1].output_last
    else:
        output_last = model.output_last

    inputs, labels = next(iter(test_dataloader))
    [batch_size, type_size, step_size, fea_size] = inputs.size()

    cur_time = time.time()
    pre_time = time.time()

    use_gpu = torch.cuda.is_available()
    
    tested_batch = 0

    losses = []
    AUROCs = []
    AUPRCs = []

    for data in test_dataloader:
        inputs, labels = data

        if inputs.shape[0] != batch_size:
            continue

        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        
        outputs = torch.nan_to_num(x=outputs, nan=np.float64(outputs.nanmean()))

        loss_valid = weighted_binary_cross_entropy(inputs=torch.squeeze(outputs), targets=torch.squeeze(labels), prevalence=0.09) 
        losses.append(float(loss_valid)) 
        
        sig_act = nn.Sigmoid()
        out_val_prob = sig_act(outputs)
        
        from torchmetrics.wrappers import BootStrapper
        from  torchmetrics import AUROC
        
        base_metric = AUROC()
        bootstrap = BootStrapper(base_metric, num_bootstraps=100)
        bootstrap.update(torch.squeeze(out_val_prob), torch.squeeze(labels).int())
        boot_output = bootstrap.compute()
        print(boot_output)
        
        auroc = BinaryAUROC()
        auroc.update(torch.squeeze(out_val_prob), torch.squeeze(labels))
        
        
        auroc_val = auroc.compute()
        auprc_val = binary_auprc(torch.squeeze(out_val_prob), torch.squeeze(labels))
        
        
        from torchmetrics.wrappers import BootStrapper
        from torchmetrics import AUROC
        
        base_metric = AUROC(task="binary")
        bootstrap = BootStrapper(base_metric, num_bootstraps=100)
        bootstrap.update(torch.squeeze(out_val_prob), torch.squeeze(labels))
        output = bootstrap.compute()
        
        AUROCs.append(float(auroc_val))
        AUPRCs.append(float(auprc_val))
        

        tested_batch += 1


    print('test loss: {}, test auroc: {}, test auprc {}'.format(np.mean(losses), 
                                                                np.mean(AUROCs), 
                                                                np.mean(AUPRCs)))
    return [losses, AUROCs, AUPRCs]


def create_sample_data(entries=10000):
    # Define the starting date and frequency
    start_date = pd.to_datetime('2023-01-01')
    freq = 'D'  # Daily frequency

    # Generate timestamps for 10000 days
    dates = pd.date_range(start_date, periods=entries, freq=freq)

    # Generate random data with some missing values
    data = np.random.rand(entries)
    # Introduce missing values randomly (around 10% missing)
    data[np.random.choice(data.shape[0], int(0.1 * len(data)))] = np.nan

    # Create a Pandas DataFrame with timestamps and data
    df = pd.DataFrame({'timestamp': dates, 'data': data})

    # Set the timestamp as the index
    df.set_index('timestamp', inplace=True)

    return df

