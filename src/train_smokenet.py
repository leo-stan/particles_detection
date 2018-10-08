#!/usr/bin/env python

""""
    File name: train_smokenet.py
    Author: Leo Stanislas
    Date created: 2018/08/06
    Python Version: 2.7
"""

import sys
sys.path.insert(0, '../model')
sys.path.insert(0, '../utils')

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from LidarDataset import LidarDataset, ToTensor
from smokenet import SmokeNet
from config import cfg
import os
from datetime import datetime

if __name__ == '__main__':

    root_dir = cfg.ROOT_DIR
    data_dir = os.path.join(root_dir, 'data')

    input_model = 'int_relpos'
    # input_model = None
    # output_model = 'test25'
    output_model = None

    writer = None
    # Control command
    batch_size = 64
    features = [
        # 'pos',
        # 'vox_pos',
        'intensity',
        # 'echo',
        'rel_pos',
    ]
    max_epochs = 25

    transform = ToTensor()

    use_gpu = True

    if use_gpu and torch.cuda.is_available():
        # Check what hardware is available
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print('Device used: ' + device.type)

    if input_model:
        input_path = os.path.join(root_dir, 'model/saved_models', input_model + '.pt')
        print('Loading model from %s' % input_path)
        batch_size, net, features, mu, sigma, _ = torch.load(input_path)
        net.to(device)
    else:
        print('Training new model')

        if output_model:
            now = datetime.now()
            log_name = output_model + '_' + now.strftime("%Y%m%d-%H%M%S")
            writer = SummaryWriter(os.path.join(root_dir, 'logs', log_name))

        epoch = 0
        id = 0
        keep_training = True
        trainset = LidarDataset(features=features, root_dir=data_dir, train=True,
                                shuffle=42, transform=transform)
        mu, sigma = trainset.mu, trainset.sigma

        t_start = time.time()

        net = SmokeNet(features_in=trainset.nb_features)
        net.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

        val_loss = None
        stop_count = 0
        print('Training...')
        try:
            while keep_training:
                trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)

                net.train()

                running_loss = 0.0
                for i_batch, sample in enumerate(trainloader):
                    # get the inputs
                    # sample = trainset[i]

                    # for s, l in zip(sample['feature_buffer'], sample['labels']):
                    inputs = sample['inputs']
                    labels = sample['labels']
                    inputs, labels = inputs.to(device), labels.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                    if i_batch % 100 == 99:  # print every 100 batches
                        if writer:
                            writer.add_scalar('Train/Loss', loss.item(), id)
                        print('Epoch: %d, Train set: %d, Running Loss: %.3f, Loss: %.3f' %
                              (epoch, id, running_loss / 100, loss.item()))
                    id += 1
                val_loss_prev = val_loss

                print('Validating...')

                valset = LidarDataset(features=features, root_dir=data_dir, train=False, val=True, mu=mu, sigma=sigma,
                                      transform=transform)

                valloader = DataLoader(valset, batch_size=int(valset.data.shape[0]/4), shuffle=False, num_workers=1)

                net.eval()
                loss = None
                with torch.no_grad():
                    for i_batch, sample in enumerate(valloader):
                        inputs = sample['inputs']
                        labels = sample['labels']
                        inputs, labels = inputs.to(device), labels.to(device)
                        # forward + backward + optimize
                        outputs = net(inputs)
                        loss = criterion(outputs, labels)
                val_loss = loss.item()
                if writer:
                    writer.add_scalar('Val/Loss', val_loss, epoch)
                if val_loss_prev:
                    print('Previous val loss: %f' % val_loss_prev)
                    print('New val loss: %f' % val_loss)
                    if val_loss_prev < val_loss:
                        stop_count += 1
                    else:
                        stop_count = 0
                if output_model:
                    torch.save([batch_size, net, features, mu, sigma, transform],
                               os.path.join(root_dir, 'logs', log_name, output_model + '_epoch' + str(epoch) + '.pt'))
                if stop_count > 4 or epoch > max_epochs:
                    keep_training = False
                else:
                    epoch += 1
        except KeyboardInterrupt:
            pass
        if writer:
            writer.close()

        print('Training finished in %f' % (time.time() - t_start))

        if output_model:
            output_path = os.path.join(root_dir, 'model/saved_models', output_model + '.pt')
            print('Saving model at %s' % output_path)
            torch.save([batch_size, net, features, mu, sigma, transform], output_path)

    print('Final evaluation on testing set...')
    correct = 0
    total = 0

    testset = LidarDataset(features=features, root_dir=data_dir, train=False, test=True, mu=mu, sigma=sigma,
                           transform=transform)

    testloader = DataLoader(testset, batch_size=int(testset.data.shape[0]/4), shuffle=False, num_workers=1)
    # testloader = DataLoader(testset, batch_size=pred_batch_size, shuffle=False, num_workers=1)

    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))

    y_pred = []
    y_test = []

    net.eval()

    with torch.no_grad():
        # for i in range(len(testset)):
        for i_batch, sample in enumerate(testloader):

            # for s, l in zip(sample['inputs'], sample['labels']):
            # s, l = s.to(device), l.to(device)

            outputs = net(sample['inputs'].to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += sample['labels'].size(0)
            correct += (predicted == sample['labels'].to(device)).sum().item()
            y_pred.append(predicted.cpu().data.numpy())
            y_test.append(sample['labels'].cpu().data.numpy())
            if i_batch % 100 == 99:  # print every 100 mini-batches
                # writer.add_scalar('Val/Loss', loss, i_batch)
                print('Test set: %d' % (i_batch + 1))

    print('Accuracy of the network on validation set: %d %%' % (
            100 * correct / total))
    # # Print network parameters
    # for f in net.parameters():
    #     print(f)

    y_pred = np.concatenate(y_pred)
    y_test = np.concatenate(y_test)

    print("Evaluating classifier...")

    print('Precision: %f' % precision_score(y_test, y_pred))

    print('Recall: %f' % recall_score(y_test, y_pred))

    print('F1 score: %f' % f1_score(y_test, y_pred))

    print('Confusion Matrix')

    cnf_matrix = confusion_matrix(y_test, y_pred)  # Compute confusion matrix
    print(cnf_matrix)

    # Can only use this if both classes are at least predicted once
    print('Classification Report')
    if len(np.unique(y_pred)) > 1:
        print(classification_report(y_test, y_pred))

    # if not input_model:
    #     writer.add_scalar('Test/Accuracy', 100 * correct / total, 0)
    #     writer.close()
