import os
import copy
from datetime import datetime

import torch
import torch.nn as nn
import torchvision
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from torch import optim
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter

import config
import apply_model
import utils
from evaluation import mean_avg_precision
from losses import OnlineTripletLoss
from metrics import AverageNonzeroTripletsMetric
from triplet_selectors import HardestNegativeTripletSelector
from data import PerClassBatchSampler
from data_transforms import PatchCrop


class TripletNetwork:

    def __init__(self, encoder, triplet_loss, pool_loss=None, losses={'triplet'}):
        self.encoder = encoder
        self.triplet_loss = triplet_loss
        self.encodings = None
        self.writers = None
        self.pool_loss = pool_loss
        self.losses = losses

    def train(self, train_data_loader, val_data_loader, num_epochs, optimizer, scheduler,
              cuda, log_interval, gmp_lambda, lr, sequential_loader, margin,
              plateau_scheduler=None, test_loader=None, metrics=[]):

        best_mAP = 0.0
        best_top1 = 0.0
        enc_best_mAP = None
        enc_best_top1 = None
        top1_best_mAP = 0
        mAP_best_top1 = 0
        writer = SummaryWriter(log_dir=config.EXPERIMENT_DIR)

        for epoch in range(num_epochs):
            if config.LR_SCHEDULER == 'exp':
                utils.adjust_lr_exp(optimizer, lr, epoch, num_epochs, config.START_EXP_DECAY)
            elif scheduler:
                scheduler.step()

            train_loss, metrics, pooling_loss = self.train_epoch(train_data_loader,
                                                   optimizer, cuda, log_interval, metrics)

            message = '\nEpoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss)
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())
                writer.add_scalar(metric.name(), metric.value(), epoch)
            if pooling_loss:
                message += '\nPooling Loss: {:.4f}'.format(pooling_loss)

            if epoch % config.VAL_INTERVAL == 0:
                mAP, top1 = self.calc_validation_metrics(sequential_loader, uses_patches=config.USE_PATCHES)
                message += '\nmAP = {}, top-1 = {}'.format(mAP, top1)
                writer.add_scalar('mAP', mAP, epoch)
                writer.add_scalar('top1', top1, epoch)

            if plateau_scheduler:
                print("step")
                plateau_scheduler.step(mAP)

            if self.encoder.pooling_type == 'gmp':
                message += '\nlambda = {}'.format(self.encoder.pool.lamb.data)
            print(message)
            print("-----------------------------")
            if mAP > best_mAP:
                best_mAP = mAP
                top1_best_mAP = top1
                enc_best_mAP = copy.deepcopy(self.encoder.state_dict())
            if top1 > best_top1:
                best_top1 = top1
                mAP_best_top1 = mAP
                enc_best_top1 = copy.deepcopy(self.encoder.state_dict())
            writer.add_scalar('loss', train_loss, epoch)

        # save models after training
        final_model = copy.deepcopy(self.encoder.state_dict())
        pooling = self.encoder.pooling_type
        self.save_final_model(pooling, final_model, gmp_lambda, lr,
                        self.encoder.__class__)

        self.save_model(pooling, enc_best_mAP, best_mAP, top1_best_mAP, lr, gmp_lambda,
                        self.encoder.__class__)
        self.save_model(pooling, enc_best_top1, mAP_best_top1, best_top1, lr, gmp_lambda,
                        self.encoder.__class__)

        if test_loader:
            print("test")
            print(config.USE_PATCHES)
            print(self.calc_validation_metrics(test_loader, documents=4, uses_patches=config.USE_PATCHES))

        # apply the best performing models to the test data
        for model in [enc_best_mAP]:  # enc_best_top1, enc_best_val_loss):
            self.encoder.load_state_dict(model)
            print("apply")
            # apply_model.evaluate(self.encoder, uses_patches=config.USE_PATCHES, color=True)
            print(self.calc_validation_metrics(test_loader, uses_patches=config.USE_PATCHES, documents=4))

    def train_epoch(self, train_data_loader, optimizer, cuda, log_interval, metrics):
        """
        Performs a training epoch.
        """
        for metric in metrics:
            metric.reset()

        self.encoder.train()
        losses = []
        pooling_losses = []
        total_loss = 0
        total_pooling_loss = 0

        for batch_idx, (data, target) in enumerate(train_data_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            optimizer.zero_grad()
            outputs = self.encoder(*data)
            # we might use the pool loss - or not.
            pool_loss = 0
            if config.LOSSES == {'triplet'}:
                descriptor = outputs
            elif config.LOSSES == {'triplet', 'pool'}:
                descriptor = outputs[0]
                unpooled = outputs[1]
                pool_loss = (0.5 * self.pool_loss(unpooled, descriptor)).item()
                pooling_losses.append(pool_loss)
            else:
                raise RuntimeError("Unsupported losses")

            # Triplet loss
            loss_inputs = (descriptor,)

            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = self.triplet_loss(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            losses.append(loss.item())

            loss += pool_loss
            total_loss += loss.item()
            total_pooling_loss += pool_loss

            loss.backward()
            optimizer.step()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

            if batch_idx % log_interval == 0:
                message = 'Train: [{}/{} ({:.0f}%)]\tTriplet Loss: {:.6f}'.format(
                    batch_idx * len(data[0]), len(train_data_loader.dataset),
                    100. * batch_idx / len(train_data_loader), np.mean(losses))
                for metric in metrics:
                    message += '\t{}: {}'.format(metric.name(), metric.value())
                if len(pooling_losses) > 0:
                    message += '\nPooling Loss: {:.4f}'.format(np.mean(pooling_losses))
                if self.encoder.pooling_type == 'gmp':
                    message += '\nlambda = {}'.format(self.encoder.pool.lamb.data)

                print(message)
                losses = []

        total_loss /= (batch_idx + 1)
        total_pooling_loss /= (batch_idx + 1)
        return total_loss, metrics, total_pooling_loss

    def test_epoch(self, test_data_loader, cuda, metrics):
        """
        Performs a test / validation epoch.
        """
        with torch.no_grad():
            for metric in metrics:
                metric.reset()
            self.encoder.eval()
            val_loss = 0
            for batch_idx, (data, target) in enumerate(test_data_loader):
                target = target if len(target) > 0 else None
                if not type(data) in (tuple, list):
                    data = (data,)
                if cuda:
                    data = tuple(d.cuda() for d in data)
                    if target is not None:
                        target = target.cuda()

                outputs = self.encoder(*data)

                if type(outputs) not in (tuple, list):
                    outputs = (outputs,)
                loss_inputs = outputs
                if target is not None:
                    target = (target,)
                    loss_inputs += target

                loss_outputs = self.triplet_loss(*loss_inputs)
                loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
                val_loss += loss.item()

                for metric in metrics:
                    metric(outputs, target, loss_outputs)

        return val_loss, metrics

    def descriptors_per_document(self, dataloader):
        """
        Computes a global descriptor for each document in the given dataset.

        Args:
            dataloader: The dataloader of the dataset,
                        should retrieve every document in an epoch exactly once.

        Returns:
            encodings: Tensor consisting of the encodings for each document
            writers: The authors corresponding to the coduments
        """
        encodings = []
        writers = []

        for data, target in dataloader:
            if torch.cuda.is_available():
                data = data.cuda()
            encoding = self.encoder(data)
            encoding = encoding.detach().cpu().numpy()
            encodings.append(encoding)
            writers.append(target.detach().cpu().numpy())

        self.encodings = np.vstack(encodings)
        self.writers = np.hstack(writers)
        return self.encodings, self.writers

    def retrieve_most_similar_documents(self, query_document, n=5):
        """
        Retrieves the top n most similar documents. The query document is represented by a
        global descriptor calculated by the encoding network. Similarity is defined by the
        euclidian distance over the encoding space.

        Args:
            query_document: global descriptor of the document.
            n: number of documents to retrieve.

        Returns:
            Top n most similar documents.
        """
        similarities = euclidean_distances(query_document.reshape(1, -1), self.encodings)
        similarities = similarities.reshape(-1)
        df = pd.DataFrame({'writer': self.writers, 'similarity': similarities})
        df = df.sort_values('similarity', ascending=True)
        df.to_csv("similarities {}".format(query_document[0]))

    def calc_validation_metrics(self, dataloader, uses_patches, documents=2):
        self.encoder.eval()
        if uses_patches:
            encodings, writers = apply_model.descriptors_for_patched_documents(
                self.encoder,
                dataloader,
                256,
                3
            )
        else:
            encodings, writers = apply_model.descriptors_per_document(self.encoder, dataloader)
        df = calculate_all_similarities(encodings, writers)
        top1 = [df[df.columns[i]][0] == df[df.columns[i]][1] for i in range(1, len(df.columns), 2)]
        top1 = sum(top1) / len(top1)
        return mean_avg_precision(df, documents, starting_col=1), top1

    def save_final_model(self, pooling, model, gmp_lambda, lr, encoder_type):
        model_dir = os.path.join('models', datetime.today().strftime('%Y-%m-%d'), pooling)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_path = os.path.join(
            model_dir,
            'final_model_gmp_lambda-{}-lr-{}_time:{}'.format(
                gmp_lambda, lr, datetime.today().strftime('%Y-%m-%d::%Hh%M')

            )
        )
        torch.save(model, model_path)
        print("wrote model to ", model_path)

    def save_model(self, pooling, model, mAP, top1, lr, gmp_lambda, encoder_type):
        model_dir = os.path.join('models', datetime.today().strftime('%Y-%m-%d'), pooling)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_path = os.path.join(
            model_dir,
            'encoder_mAP-{}_top1-{}_gmp_lambda-{}-lr-{}_time:{}'.format(
                mAP, top1, gmp_lambda, lr, datetime.today().strftime('%Y-%m-%d::%Hh%M')

            )
        )
        torch.save(model, model_path)
        print("wrote model to ", model_path)


def calculate_all_similarities(encodings, targets):
    # calculate distance matrix
    # i-th row corresponds to ith sample
    dist_matrix = euclidean_distances(encodings, squared=True)
    df = pd.DataFrame()
    for i in range(dist_matrix.shape[0]):
        query_name = 'q{} dist'.format(i)
        query_targets = 'q{} classes'.format(i)
        df[query_name] = dist_matrix[i]
        df[query_targets] = targets
        df[query_targets] = df[query_targets].astype('int32')
    # sort associated columns in the data frame
    for i in range(0, len(df.columns), 2):
        s = df[df.columns[i: i+2]]
        s = s.sort_values(df.columns[i])
        df[df.columns[i: i + 2]] = s.values
    return df


class Trainer:

    def __init__(self, data, rankings_output, optimizer, scheduler, num_classes, num_samples, encoder: nn.Module):
        self.data = data
        self.rankings_output = rankings_output
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.encoder = encoder

    def train_net(self, num_epochs, lr, triplet_selector, gmp_lambda, margin=1.,
                  scheduler_step=config.SCHEDULER_STEP, lr_decay=config.LR_DECAY,
                  weight_decay=config.WEIGHT_DECAY, log_interval=50):


        # TODO: different LR for different layers
        # parameter_options = [
        #     {'params': encoder.model.layer1.parameters(), 'lr': lr / 1000},
        #     {'params': encoder.model.layer2.parameters(), 'lr': lr / 100},
        #     {'params': encoder.model.layer3.parameters(), 'lr': lr / 10},
        #     {'params': encoder.model.layer4.parameters(), 'lr': lr / 10},
        #     {'params': encoder.model.avgpool.parameters(), 'lr': lr},
        #     {'params': encoder.model.fc.parameters(), 'lr': lr}
        # ]
        parameters = [
            {'params': self.encoder.feature_extractor.parameters()},
            {'params': self.encoder.pool.parameters(), 'lr': config.POOL_LR_MULTIPLIER * lr, 'weight_decay': 0}
        ]

        if self.optimizer == 'adam' or self.optimizer =='Adam':
            optimizer = optim.Adam(parameters, lr, weight_decay=weight_decay, amsgrad=True)
        elif self.optimizer in ('sgd', 'SGD'):
            optimizer = optim.SGD(parameters, lr, momentum=0.9,
                                  weight_decay=weight_decay, nesterov=True)

        scheduler = None
        plateau_scheduler = None
        if self.scheduler == 'step-lr':
            scheduler = lr_scheduler.StepLR(optimizer, scheduler_step, lr_decay)
            plateau_scheduler = None
        elif self.scheduler == 'plateau':
            scheduler = None
            plateau_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 0.1,
                                                               patience=scheduler_step, verbose=True)

        train_sampler = PerClassBatchSampler(self.data.train_data,
                                             num_classes=self.num_classes,
                                             num_samples=self.num_samples)
        val_sampler = PerClassBatchSampler(self.data.val_data,
                                           num_classes=self.num_classes,
                                           num_samples=self.num_samples)
        train_data_loader = self.data.get_train_data_loader(batch_sampler=train_sampler)
        val_loader = self.data.get_val_data_loader(batch_sampler=val_sampler)

        cuda = torch.cuda.is_available()
        if cuda:
            self.encoder.cuda()

        loss_fn = OnlineTripletLoss(margin, triplet_selector)

        triplet_net = TripletNetwork(self.encoder, triplet_loss=loss_fn)

        if config.USE_PATCHES:
            trans = torchvision.transforms.Compose([
                PatchCrop(400, resize_to=256),
                # torchvision.transforms.Resize(256),
                torchvision.transforms.Lambda(
                    lambda crops: [torchvision.transforms.ToTensor()(crop) for crop in crops]),
            ])
            seq_loader = self.data.get_val_data_loader2(transform=trans)
            test_loader = self.data.get_test_data_loader(transform=trans)
        else:
            seq_loader = self.data.get_sequential_data_loader(batch_size=15)
            test_loader = self.data.get_competition_data_loader()

        triplet_net.train(train_data_loader, val_loader, num_epochs,
                          optimizer, scheduler, cuda, log_interval,
                          gmp_lambda, lr, seq_loader, margin,
                          test_loader=test_loader, plateau_scheduler=plateau_scheduler,
                          metrics=[AverageNonzeroTripletsMetric(),])

    def find_lr(self, lr_start, lr_end, lr_multiplier, gmp_lambda, num_epochs, margin,
                scheduler_step, lr_decay, weight_decay):
        lr = lr_start
        while lr > lr_end:
            triplet_selector = HardestNegativeTripletSelector(margin)
            self.train_net(num_epochs, lr, triplet_selector, gmp_lambda, margin,
                           scheduler_step, lr_decay, weight_decay)
            lr = lr * lr_multiplier

    def find_lr_and_lambda(self, lr_start, lr_end, lr_multiplier, num_epochs, margin,
                           scheduler_step, lr_decay, weight_decay, lambda_start, lambda_end,
                           lambda_multiplier):
        lr = lr_start
        while lr > lr_end:
            print('******\n++ find lr and lambda. lr = {} ++'.format(lr))
            self.find_lambda(lambda_start, lambda_end, lambda_multiplier, margin, lr, num_epochs,
                             scheduler_step, lr_decay, weight_decay)
            lr = lr * lr_multiplier

    def find_lambda(self, lambda_start, lambda_end, lambda_multiplier, margin, lr, num_epochs,
                    scheduler_step, lr_decay, weight_decay):

        gmp_lambda = lambda_start
        while gmp_lambda < lambda_end:
            if not margin:
                triplet_selector = HardestNegativeTripletSelector(0)
            else:
                triplet_selector = HardestNegativeTripletSelector(margin)
            print('find lambda= {}, lr = {}'.format(gmp_lambda, lr))
            self.train_net(num_epochs, lr, triplet_selector, gmp_lambda, margin, scheduler_step=scheduler_step,
                           lr_decay=lr_decay, weight_decay=weight_decay)
            gmp_lambda = gmp_lambda * lambda_multiplier