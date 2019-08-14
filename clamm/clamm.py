"""
Evaluate GMP for clamm16/clamm17 
author: Vincent Christlein
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, transforms
import datetime
import time
import os
import copy
import dataset_label
from util import *
from init_model import initialize_model, set_grads
import configargparse
from tensorboardX import SummaryWriter
import glob

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


######################################################################
# Inputs
# ------
def parseArgs(parser):
    parser.add('expdir', 
               help='experiment directory and identifier name')
    parser.add('-c', '--config', is_config_file=True, 
               help='config file path')
    # data
    parser.add('--traindir',
               help='the input folder of the training images / features')
    parser.add('--train_labels', 
               help='contains training images/descriptors to load + labels')
    parser.add('--valdir',
               help='the input folder of the val images / features')
    parser.add('--val_labels', 
               help='contains val images/descriptors to load + labels')
    parser.add('--testdir',
               help='the input folder of the test images / features')
    parser.add('--test_labels', 
               help='contains test images/descriptors to load + labels')
    parser.add('-s', '--suffix',
               default='_SIFT_patch_pr.pkl.gz',
               help='only chose those images with a specific suffix')
    parser.add('-j', '--workers', default=4, type=int, metavar='N',
               help='number of data loading workers (default: 4)')

    # model 
    parser.add('--model_name', default='poolnet',
               choices=['poolnet', 'resnet18','resnet50'],
               help='model architecture')
    parser.add('--last_conv_stride', type=int, default=2,
               help='set last conv stride for resnet')
    parser.add('--input_size', type=int, default=384,
               help='input image/patch size')
    parser.add('--pool_type', default='avg', 
               choices=['avg', 'max', 'gmp',  'weighted_avg', 
                        'gem', 'conv', 'max', 'lse', 'mixed-pool', 'gmp-fix',
                        'gemaxmean'],
               help='pooling type')
    parser.add('--pool_lr_multiplier', default=0.0, type=float,
               help='pooling learning rate multiplier')    
    parser.add('--class_lr_multiplier', default=0.0, type=float,
               help='pooling learning rate multiplier')    
    parser.add('--gmp_lambda', type=float, default=1000.0,
               help='generalized max pooling lambda parameter')
    parser.add('--lse_r', type=float, default=10.0,
               help='lse parameter')
    parser.add('--n_classes', type=int, default=12,
               help='number of classes')
    parser.add('--finetune', action='store_true',
               help='use a pre-trained model')
    parser.add('--finetune_start', type=int, default=0,
               help='start finetuning whole network at this epoch')
    parser.add('-b', '--batch_size', default=32, type=int,
               metavar='N', help='mini-batch size')
    parser.add('--batch_size_val', default=32, type=int,
               metavar='N', help='mini-batch size validation')
    parser.add('--batch_size_test', default=64, type=int,
               metavar='N', help='mini-batch size test')
   
    # optimizer
    parser.add('--config_optim', is_config_file=True, 
               help='config file path')
    parser.add('--optimizer', default='sgd', choices=['sgd', 'adam'],
               help='type of optimizer')
    parser.add('--lr', default=0.01, type=float,
               help='learning rate')
    parser.add('--lr_scheduler', default='exp', 
               help='1. <exp> for exponential decay, '
               'starting at start_exp_decay, 2. <steplr:step_size:gamma>,'
               'where step_size is the epoch number and gamma is the'
               'learning rate multiplier or 3. <plateau>')
    parser.add('--start_exp_decay', default=40, type=int,
               help='when to start exponential decay')
    parser.add('--n_epochs', default=140, type=int, 
               help='number of total epochs to run')
    parser.add('--start_epoch', default=0, type=int, 
               help='manual epoch number (useful on restarts)')
    
    parser.add('--momentum', default=0.9, type=float,
               help='momentum')
    parser.add('--nesterov', action='store_true', default=False,
               help='use nesterov momentum')
    parser.add('--weight_decay', '--wd', default=1e-4, type=float,
               help='weight decay')
    # log
    parser.add('--check_epoch', type=int, default=50,
               help='create checkpoint every x epoch')
    parser.add('--print_freq', type=int, default=1,
               help='print frequency for log/print')
    parser.add('--resume', type=str, metavar='PATH', 
               help='path to latest checkpoint,'
               ' default==expdir/checkpoints/best_model.pth.tar')    
    # misc
    parser.add('--test_only', action='store_true',
               help='test only with a given model')
    parser.add('--seed', type=int, default=42,
               help='set seed')
    parser.add('--normalize', choices=['l2','none'], default='l2', 
               help='normalize the feature extractor (after pooling) for'
               ' poolnet')
    parser.add('--ensemble', default=[], nargs='*', 
               help='all model paths for which ensemble should be computed')
    return parser



# Model Training and Validation Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def run_epoch(model, dataloader, phase, optimizer, criterion, epoch, 
              device, compute_extra=False):
    # TODO: convert running_loss and running_corrects to AverageMeter()
    running_loss = 0.0
    running_corrects = 0
    all_inputs = 0
    batch_time = AverageMeter()
    data_time = AverageMeter()
    model_time = AverageMeter()
    
    # for majority vote 
    all_pred = []
    # for sum voting of the logits
    out_sum = None
    
    all_outputs = []

    end = time.time()
    for e, (inputs, labels) in enumerate(dataloader):
        data_time.update(time.time() - end)
        
        # should also be at the cuda device due to pin_memory, or?
        # but a second call shouldn't matter
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        with torch.set_grad_enabled(phase == 'train'):
            # Get model outputs and calculate loss
            if inputs.ndimension() == 5:
                bs, ncrops, c, h, w = inputs.size()
                result = model(inputs.view(-1, c, h, w))
                outputs = result.view(bs, ncrops, -1).mean(1)
            else:
                outputs = model(inputs)
            model_time.update(time.time()-end)

            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            # future comp
            if compute_extra:
                output_sum = torch.sum(outputs, dim=0, keepdim=True)
                out_sum = output_sum if out_sum is None else out_sum + output_sum
                all_pred.append(preds)
                all_outputs.append(outputs.cpu())

            # backward + optimize only if in training phase
            if phase == 'train':
                loss.backward()
                optimizer.step()
       
        # AverageMeter?
        corrects = torch.sum(preds == labels.data)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += corrects.double()
        all_inputs += inputs.size(0)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()    

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    print('[{0}: {1}]\t'
          'Time {batch_time.sum:.3f} ({batch_time.avg:.3f})\t'
          'Data {data_time.sum:.3f} ({data_time.avg:.3f})\t'
          'Model {model_time.sum:.3f} ({model_time.avg:.3f})\t'
          'Loss {running_loss:.4f} ({epoch_loss:.4f})\t'
          'Prec@1 {running_corrects:.3f} ({epoch_acc:.3f})\t'\
          .format(
              phase, epoch, batch_time=batch_time,
              model_time=model_time,
              data_time=data_time, 
              running_loss=running_loss, epoch_loss=epoch_loss,
              running_corrects=running_corrects.double(),
              epoch_acc=epoch_acc
          ))
    
    if compute_extra:
        return running_loss, running_corrects, epoch_loss, epoch_acc, \
                out_sum, all_pred, torch.cat(all_outputs)

    return running_loss, running_corrects, epoch_loss, epoch_acc

def train_model(model, dataloaders, criterion, optimizer, 
                start_epoch=0, n_epochs=25, 
                finetune_start=0,
                lr=0.01, lr_scheduler=None, start_exp_decay=40,
                print_freq=1, 
                expdir='experiments', check_epoch=25, device=0,
                best_acc=0, writer=None):
    since = time.time()
    checkpoint_path = os.path.join(expdir,'checkpoints')
    if not os.path.exists(checkpoint_path):
        mkdir_p(checkpoint_path)
    
    best_model = copy.deepcopy(model.state_dict())

    val_log = open(os.path.join(expdir,'val.log'), 'w' if start_epoch == 0 else\
                   'a', buffering=1)
    if start_epoch == 0:
        val_log.write('epoch,loss,acc\n')
    train_log = open(os.path.join(expdir,'train.log'), 'w' if start_epoch == 0\
                     else 'a', buffering=1)
    if start_epoch == 0:
        train_log.write('epoch,loss,acc\n')
    for epoch in range(start_epoch, n_epochs):        
        full_epoch_time = time.time()
        print('Epoch {}/{}'.format(epoch, n_epochs - 1))
        print('-' * 10)
        
        if epoch == finetune_start and finetune_start > 0:
            print('now train full model')
            old_non_grads,_ = set_grads(model, requires_grad=True)
            optimizer.add_param_group({'params':old_non_grads})
       
        # this scheduler just depends on the epoch
        if lr_scheduler == 'exp':
            adjust_lr_exp(optimizer, lr, epoch, n_epochs,
                          start_exp_decay)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss, running_corrects, epoch_loss, epoch_acc = run_epoch(model,
                                                       dataloaders[phase], phase,
                                                       optimizer, criterion,
                                                       epoch, 
                                                       device)

#            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            n_iter = epoch # * len(dataloaders['train'].dataset)
            if phase == 'val':
                writer.add_scalar('Test/Loss', epoch_loss, n_iter)
                writer.add_scalar('Test/Prec@1', epoch_acc, n_iter)
                val_log.write('{},{},{}\n'.format(epoch+1,epoch_loss,epoch_acc))
            else:
                writer.add_scalar('Train/Loss', epoch_loss, n_iter)
                writer.add_scalar('Train/Prec@1', epoch_acc, n_iter)
                train_log.write('{},{},{}\n'.format(epoch+1,epoch_loss,epoch_acc))
            
            is_best = False
            if phase == 'val' and epoch_acc > best_acc:
                is_best = True
                best_acc = epoch_acc
                best_model = copy.deepcopy(model.state_dict())

            # deep copy the model
            if phase == 'val':
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                    }, checkpoint_path=checkpoint_path, 
                    is_best=is_best, 
                    epoch=epoch, check_epoch=check_epoch
                )

        epoch_time = time.time() - full_epoch_time
        eta = (n_epochs-epoch) * epoch_time
        print('needed {:.0f}s/Epoch : ETA: ~{:.0f}m'.format(epoch_time,
                                                            eta // 60))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    train_log.close()
    val_log.close()                                

    # load best model weights
    model.load_state_dict(best_model)
    return model

def main():

    parser = configargparse.ArgParser(description='PyTorch Training',
                                      formatter_class=configargparse.ArgumentDefaultsRawHelpFormatter)
    parser = parseArgs(parser)
    args = parser.parse_args()
    print('{}'.format(args))


#   TODO: we would need to fix a lot of other seeds too!
#    torch.manual_seed(args.seed)
#   --> Currently not reproducible runs, but
#   since we use multiple runs and average, the results should be close
#   to the ones reported in the paper

    if os.path.isdir(args.expdir) and not args.test_only:
        print('WARNING: path already exists!')
        for i in range(2,99):
            path = args.expdir + str(i)
            if not os.path.isdir(path): 
                args.expdir = path
                print('set new expdir:', path)
                break
    
    if args.ensemble:
        args.test_only = True

    if not os.path.isdir(args.expdir):
        mkdir_p(args.expdir)
    writer = SummaryWriter(os.path.join(args.expdir,'runs'))
    if not args.test_only:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
        cmd_log = open(os.path.join(args.expdir, 'cmd.log'), 'a')
        cmd_log.write('{}\n'.format(timestamp))
        cmd_log.write('{}\n'.format(args))
        cmd_log.write(parser.format_values())
    start_time = time.time()
    if args.resume is not None and args.resume == 'default':
        args.resume = os.path.join(args.expdir, 'checkpoints', 'best_model.pth.tar')
        print('try to resume: {}'.format(args.resume))
    
    # Initialize the model for this run
    model = initialize_model(args.model_name, 
                             args.n_classes, 
                             use_pretrained=args.finetune,
                             pool_type=args.pool_type,
                             gmp_lambda=args.gmp_lambda,
                             lse_r=args.lse_r,
                             last_conv_stride=args.last_conv_stride,
                             normalize=args.normalize)
    
    # Print the model we just instantiated
    print(model) 
    
    ### Data augmentation and normalization for training ###
    
    # pxl mean: 185.2293097, std: 61.8039951063
    m = 185.2293097 / 255
    s = 61.8039951063 / 255  
    
    data_transforms = {
            'train': transforms.Compose([
                # TODO: maybe more random scale?
                transforms.RandomResizedCrop(args.input_size),
                transforms.RandomRotation(degrees=5),
                # what's with hue and saturation as with the patch-wise?
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize([m,m,m], [s, s, s])
            ]),
            #'val': #test_transform
            'val': transforms.Compose([
                transforms.Resize(args.input_size),
                transforms.CenterCrop(args.input_size),
                transforms.ToTensor(),
                transforms.Normalize([m, m, m], [s, s, s])
            ]),
        }
    
    print("Initializing Datasets and Dataloaders...")
    
    data_dir = {}
    data_dir['train'] = args.traindir
    data_dir['val'] = args.traindir if not args.valdir else args.valdir
    labels = {}
    labels['train'] = args.train_labels
    labels['val']   = args.val_labels
    
    # Create training and validation datasets
    image_datasets = {x: dataset_label.DatasetLabel(data_dir[x], labels[x],
                                                    args.suffix, 
                                                    transform=data_transforms[x]) \
                      for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                       batch_size=args.batch_size\
                                                        if x == 'train' else\
                                                       args.batch_size_val,
                                                       shuffle=True if \
                                                           x == 'train' else False,
                                                       num_workers=args.workers,
                                                       pin_memory=False, 
                                                       drop_last=True) \
                        for x in ['train', 'val']}
    
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Send the model to GPU (has to be done before constructing optim 
    model = model.to(device)
    
    # Observe that all parameters are being optimized
    if not args.finetune: 
        # we have to train the full model from scratch
        # with this we set all parameters for update
        finetune_start = 0 
    else:
        finetune_start = args.finetune_start

    assert(finetune_start == 0) 
 
    print('set all remaining parameters to requires_grad=True')
    if args.model_name == 'poolnet':
        fe_params = {'params': model.feature_extractor.parameters()}
        # shall we treat the pool layer differently?
        # --> doesnt seem to be needed, or?
        if args.pool_lr_multiplier != 0:
            pool_params = {'params': model.pool.parameters(), 
                           'lr': args.pool_lr_multiplier * args.lr, 
                           'weight_decay': 0}
        else:
            pool_params = {'params': model.pool.parameters()}
        
        if args.class_lr_multiplier != 0: 
            fc_params = {'params': model.fc.parameters(),
                         'lr': args.class_lr_multiplier * args.lr}
        else:
            fc_params = {'params': model.fc.parameters()}
        
        params_to_update = [fe_params, pool_params, fc_params] 
    else:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

    if args.optimizer == 'adam': 
        optimizer = optim.Adam(params_to_update, lr=args.lr, 
                               weight_decay=args.weight_decay, 
                               amsgrad=True)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(params_to_update, lr=args.lr, 
                              momentum=args.momentum,
                              weight_decay=args.weight_decay,
                              nesterov=args.nesterov)
    else:
        raise ValueError('unknown optimizer: {}'.format(args.optimizer))
  
    if args.normalize == 'learn':
        optimizer.add_param_group({'params':model.normalize.parameters()})

    best_acc = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            model.to(device)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(args.resume))
    
    
    ######################################################################
    # Run Training and Validation Step
    # --------------------------------
    # 
    # Finally, the last step is to setup the loss for the model, then run the
    # training and validation function for the set number of epochs. Notice,
    # depending on the number of epochs this step may take a while on a CPU.
    # Also, the default learning rate is not optimal for all of the models, so
    # to achieve maximum accuracy it would be necessary to tune for each model
    # separately.
    # 
    
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
    
    # Train and evaluate
    if not args.test_only:
        model = train_model(model, dataloaders_dict, criterion,
                            optimizer, start_epoch=args.start_epoch,
                            n_epochs=args.n_epochs,
                            finetune_start=finetune_start,
                            lr=args.lr, lr_scheduler=args.lr_scheduler, 
                            start_exp_decay=args.start_exp_decay,
                            print_freq=args.print_freq,
                            expdir=args.expdir,
                            check_epoch=args.check_epoch,
                            device=device,
                            best_acc=best_acc, writer=writer)
    
    model.eval()
    # test model
    test_dataset = dataset_label.DatasetLabel(args.testdir, args.test_labels,
                                              args.suffix, 
                                              transform=data_transforms['val']) 

    test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                                  batch_size=args.batch_size, 
                                                  shuffle=False, 
                                                  num_workers=args.workers, 
                                                  pin_memory=True) 
    if len(args.ensemble) > 0:
        if '*' in args.ensemble[0] or '?' in args.ensemble[0]:
            ensemble = glob.glob(args.ensemble[0])
        else:
            ensemble = args.ensemble
        assert(len(ensemble) > 0)

        print('ensemble of:', ensemble)

        all_pred = []
        all_outputs = []
        all_best_val = []
        all_test_acc = []
        for e,ens_file in enumerate(ensemble):
            print('ensemble {} / {}'.format(e+1,len(ensemble)))

            if os.path.isfile(ens_file):
                print("=> loading ensemble checkpoint '{}'".format(ens_file))
                checkpoint = torch.load(ens_file)
                all_best_val.append(checkpoint['best_acc'].cpu().numpy())
                model.load_state_dict(checkpoint['state_dict'])
                model.to(device)
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded ensemble checkpoint '{}' (epoch {})"
                      .format(ens_file, checkpoint['epoch']))
            else:
                raise ValueError('=> no checkpoint found at'
                                 ' {}'.format(ens_file))

            _, _, _, test_acc, _, pred, outputs = run_epoch(model, test_dataloader, 'test', 
                                          optimizer, criterion, 0,
                                          device, compute_extra=True)
            all_test_acc.append(test_acc)
            all_pred.append(torch.cat(pred).cpu().reshape(-1,1))
            all_outputs.append(outputs)
            print('single test accuracy: {}'.format(test_acc))
            with open(os.path.join(args.expdir, 'test.log'), 'a') as f:
                f.write('acc: {}\n'.format(test_acc))

        np_labels = np.array(list(zip(*test_dataset.samples))[1])
        labels = torch.from_numpy(np_labels)
        
        # all predictions of all classifiers
        all_pred = torch.cat(all_pred, dim=1)
        # now let's take majority
        maj, _ = all_pred.mode(dim=1)
        maj_acc = torch.sum(maj == labels)
        maj_acc = maj_acc.double() / float(len(test_dataset))
    
        print('majority test accuracy: {}'.format(maj_acc))
        with open(os.path.join(args.expdir, 'maj.log'), 'a') as f:
            f.write('acc: {}\n'.format(maj_acc))
        
        # via softmax
        all_outputs = torch.stack(all_outputs) # n_ensembles x N x C
        soft = nn.functional.softmax(all_outputs, 2)
        new_out = torch.sum(soft, 0) # -> N x C
        _, preds = torch.max(new_out, 1)

        soft_acc = torch.sum(preds == labels).double() / float(len(test_dataset))
        print('softmax test accuracy: {}'.format(soft_acc))
        with open(os.path.join(args.expdir, 'soft.log'), 'a') as f:
            f.write('soft: {}\n'.format(soft_acc))

        # weighted softmax
        all_best_val = np.array(all_best_val)
        all_best_val /= all_best_val.sum() # l1 norm
        for i in range(len(all_best_val)):
            soft[i,:,:] *= all_best_val[i]
        new_out = torch.sum(soft, 0) # -> N x C
        _, preds = torch.max(new_out, 1)

        wsoft_acc = torch.sum(preds == labels).double() / len(test_dataset)
        print('weighted softmax test accuracy: {}'.format(wsoft_acc))
        with open(os.path.join(args.expdir, 'wsoft.log'), 'a') as f:
            f.write('wsoft: {}\n'.format(wsoft_acc))
        
    else:
        _, _, _, test_acc = run_epoch(model, test_dataloader, 'test', 
                                                     optimizer, criterion, 0,
                                                     #eval_stats, 
                                                     device)
        print('test accuracy: {}'.format(test_acc))
        with open(os.path.join(args.expdir, 'test.log'), 'a') as f:
            f.write('acc: {}\n'.format(test_acc))            

    if not args.test_only:
        curr_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
        cmd_log.write('Finished: {}\n'.format(curr_time))
        time_elapsed = time.time() - start_time
        cmd_log.write('Duration {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
        cmd_log.close()

if __name__ == '__main__':
    main()
