#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_resnet.py: This file contains function to train
the Resnet model. 
"""

__author__ = "Duret Jarod, Brignatz Vincent"
__license__ = "MIT"

import time
import numpy as np

from tqdm import tqdm
from loguru import logger
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import dataset
from models import resnet34, NeuralNetAMSM, ContrastLayer
from test_resnet import score_utt_utt, score_contrastive

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

@logger.catch
def train(args, generator, classifier, dataloader_train, device, dataset_validation=None):
    # Tensorflow logger
    writer = SummaryWriter(comment='_{}'.format(args.model_dir.name))

    # Load the trained model if we continue from a checkpoint
    start_iteration = 0
    if args.checkpoint > 0:
        start_iteration = args.checkpoint
        for model, modelstr in [(generator, 'g'), (classifier, 'c')]:
            model.load_state_dict(torch.load(args.checkpoints_dir / f'{modelstr}_{args.checkpoint}.pt'))
    
    elif args.checkpoint == -1:
        start_iteration = max([int(filename.stem[2:]) for filename in args.checkpoints_dir.iterdir()])
        for model, modelstr in [(generator, 'g'), (classifier, 'c')]:
            model.load_state_dict(torch.load(args.checkpoints_dir / f'{modelstr}_{start_iteration}.pt'))

    # Optimizer definition
    optimizer = torch.optim.SGD([{'params': generator.parameters(), 'lr': args.generator_lr},
                                 {'params': classifier.parameters(), 'lr': args.classifier_lr}],
                                momentum=args.momentum)

    criterion = nn.CrossEntropyLoss()

    # multi GPU support :
    if args.multi_gpu:
        dpp_generator = nn.DataParallel(generator).to(device)
    
    if dataset_validation is not None:
        best_eer = {'eer':100, 'ite':-1}

    start = time.process_time()
    for iterations in range(start_iteration, args.num_iterations + 1):
        # The current iteration is specified in the scheduler
        # Reduce the learning rate by the given factor (args.scheduler_lambda)
        if iterations in args.scheduler_steps:
            for params in optimizer.param_groups:
                params['lr'] *= args.scheduler_lambda
            print(optimizer)

        avg_loss = 0
        for feats, spk, utt in dataloader_train:
            feats = feats.unsqueeze(1).to(device)
            spk = torch.LongTensor(spk).to(device)

            # Creating embeddings
            if args.multi_gpu:
                embeds = dpp_generator(feats)
            else:
                embeds = generator(feats)

            # Classify embeddings
            preds = classifier(embeds, spk)

            # Calc the loss
            loss = criterion(preds, spk)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        
        avg_loss /= len(dataloader_train)
        # Write the loss in tensorflow
        writer.add_scalar('Loss', avg_loss, iterations)

        # loguru logging :
        if iterations % args.log_interval == 0:
            msg = "{}: {}: [{}/{}] \t C-Loss:{:.4f}, lr: {}, bs: {}".format(args.model_dir,
                                                                            time.ctime(),
                                                                            iterations,
                                                                            args.num_iterations,
                                                                            avg_loss,
                                                                            get_lr(optimizer),
                                                                            args.batch_size
                                                                            )
            logger.info(msg) 

         # Saving checkpoint
        if iterations % args.checkpoint_interval == 0:
            
            for model, modelstr in [(generator, 'g'), (classifier, 'c')]:
                model.eval().cpu()
                cp_model_path = args.checkpoints_dir / f"{modelstr}_{iterations}.pt"
                torch.save(model.state_dict(), cp_model_path)
                model.to(device).train()

            # Testing the saved model
            if dataset_validation is not None:
                logger.info('Model Evaluation')
                eer = score_utt_utt(generator, dataset_validation)['eer']
                logger.info(f'EER : {eer}')
                writer.add_scalar(f'EER', eer, iterations)
                if eer < best_eer["eer"]:
                    best_eer["eer"] = eer
                    best_eer["ite"] = iterations
                logger.success(f"\nBest score is at iteration {best_eer['ite']} : {best_eer['eer']} eer")
            logger.info(f"Saved checkpoint at iteration {iterations}")

    # Final model saving
    for model, modelstr in [(generator, 'g'), (classifier, 'c')]:
        model.eval().cpu()
        cp_filename = "final_{}_{}.pt".format(modelstr, iterations)
        cp_model_path = args.model_dir / cp_filename
        torch.save(model.state_dict(), cp_model_path)
    logger.success(f'Training complete in {time.process_time()-start} seconds')

@logger.catch
def train_contrastive(args, generator, projection_head, dataloader_train, device, dataset_validation=None):
    # Tensorflow logger
    writer = SummaryWriter(comment='_{}'.format(args.model_dir.name))

    # Load the trained model if we continue from a checkpoint
    start_iteration = 0
    if args.checkpoint > 0:
        start_iteration = args.checkpoint
        generator.load_state_dict(torch.load(args.checkpoints_dir / f'g_{args.checkpoint}.pt'))
    
    # elif args.checkpoint == -1:
    #     start_iteration = max([int(filename.stem[2:]) for filename in args.checkpoints_dir.iterdir()])
    #     for model, modelstr in [(generator, 'g'), (classifier, 'c')]:
    #         model.load_state_dict(torch.load(args.checkpoints_dir / f'{modelstr}_{start_iteration}.pt'))

    # Optimizer definition
    optimizer = torch.optim.SGD([{'params': generator.parameters(), 'lr': args.generator_lr},
                                 {'params': projection_head.parameters(), 'lr': args.classifier_lr}],
                                momentum=args.momentum)

    criterion = nn.CrossEntropyLoss()

    # multi GPU support :
    if args.multi_gpu:
        dpp_generator = nn.DataParallel(generator).to(device)
    
    if dataset_validation is not None:
        best_eer = {'eer':100, 'ite':-1}

    start = time.process_time()
    for iterations in range(start_iteration, args.num_iterations + 1):
        # The current iteration is specified in the scheduler
        # Reduce the learning rate by the given factor (args.scheduler_lambda)
        if iterations in args.scheduler_steps:
            for params in optimizer.param_groups:
                params['lr'] *= args.scheduler_lambda
            print(optimizer)

        avg_loss = 0
        for f1, f2, spk in dataloader_train:
            f1 = f1.unsqueeze(1)
            f2 = f2.unsqueeze(1)
            features = torch.cat((f1, f2), dim=0).to(device)

            # Creating embeddings
            if args.multi_gpu:
                embeds = dpp_generator(features)
            else:
                embeds = generator(features)
            embeds = projection_head(embeds)

            # Labels calculation
            labels = torch.cat([spk, spk], dim=0)
            labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
            labels = labels.to(device)

            # Cosine calculation
            embeds = F.normalize(embeds, dim=1)
            similarity_matrix = torch.matmul(embeds, embeds.T)

            # discard the main diagonal from both: labels and similarities matrix
            mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
            labels = labels[~mask].view(labels.shape[0], -1)
            similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

            # select and combine multiple positives
            positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

            # select only the negatives the negatives
            negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

            logits = torch.cat([positives, negatives], dim=1)
            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

            # Calc the loss
            loss = criterion(logits, labels)

            # Backpropagationclassifier
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            avg_loss += loss.item()
        
        avg_loss /= len(dataloader_train)
        # Write the loss in tensorflow
        writer.add_scalar('Loss', avg_loss, iterations)

        # loguru logging :
        if iterations % args.log_interval == 0:
            msg = "{}: {}: [{}/{}] \t C-Loss:{:.4f}, lr: {}, bs: {}".format(args.model_dir,
                                                                            time.ctime(),
                                                                            iterations,
                                                                            args.num_iterations,
                                                                            avg_loss,
                                                                            get_lr(optimizer),
                                                                            args.batch_size
                                                                            )
            logger.info(msg) 

         # Saving checkpoint
        if iterations % args.checkpoint_interval == 0:
            
            for model, modelstr in [(generator, 'g'), (projection_head, 'c')]:
                model.eval().cpu()
                cp_model_path = args.checkpoints_dir / f"{modelstr}_{iterations}.pt"
                torch.save(model.state_dict(), cp_model_path)
                model.to(device).train()

            # Testing the saved model
            if dataset_validation is not None:
                logger.info('Model Evaluation')
                eer = score_contrastive(generator, projection_head, dataset_validation)['eer']
                logger.info(f'EER : {eer}')
                writer.add_scalar(f'EER', eer, iterations)
                if eer < best_eer["eer"]:
                    best_eer["eer"] = eer
                    best_eer["ite"] = iterations
                logger.success(f"\nBest score is at iteration {best_eer['ite']} : {best_eer['eer']} eer")
            logger.info(f"Saved checkpoint at iteration {iterations}")

    # Final model saving
    for model, modelstr in [(generator, 'g'), (projection_head, 'c')]:
        model.eval().cpu()
        cp_filename = "final_{}_{}.pt".format(modelstr, iterations)
        cp_model_path = args.model_dir / cp_filename
        torch.save(model.state_dict(), cp_model_path)
    logger.success(f'Training complete in {time.process_time()-start} seconds')