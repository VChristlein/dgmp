import torch
import shutil
import sys
import shlex
import os
import gzip
if sys.version_info >= (3, 0):
    import pickle
    import _pickle as cPickle
else:
    import cPickle
from collections import OrderedDict
from torch.optim.optimizer import Optimizer

class AverageMeter(object):
  """
  Computes and stores the average and current value
  """
  def __init__(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val*n
    self.count += n
    self.avg = self.sum/self.count


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

def save_checkpoint(state, checkpoint_path='checkpoint', 
                    is_best=False, epoch=0,
                    check_epoch=1, is_last=False):
    # normal save        
    filename = 'modelbak_' + ('even' if epoch % 2 == 0 else 'odd')
    path = os.path.join(checkpoint_path, filename) + '.pth.tar'
    torch.save(state, path)
    if is_best:
        shutil.copy(path, os.path.join(checkpoint_path, 'best_model.pth.tar'))
    if is_last:
        shutil.copy(path, os.path.join(checkpoint_path, 'final_model.pth.tar'))
    if epoch % check_epoch == 0 and epoch > 0:
        shutil.copy(path, os.path.join(checkpoint_path,
                                       'model_{}.pth.tar'.format(epoch+1)))

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def getFiles(folder, pattern, labelfile):
    """ 
    returns files and associated labels by reading the labelfile 
    parameters:
        folder: inputfolder
        pattern: new suffix
        labelfiles: contains a list of filename and labels
    return: absolute filenames + labels 
    """
    assert(pattern)

    # read labelfile
    if not labelfile:
        import glob
        all_files = glob.glob(os.path.join(folder, '*' + pattern))
        return all_files, None

    with open(labelfile, 'r') as f:
        all_lines = f.readlines()
    
    # get filenames from labelfile
    all_files = []
    labels = []
#    check = True
    for line in all_lines:
        # using shlex we also allow spaces in filenames when escaped w. ""
        splits = shlex.split(line)
        file_name = splits[0]
        class_id = splits[1]

        # strip all known endings, note: os.path.splitext() doesnt work for
        # '.' in the filenames, so let's do it this way...
        for p in ['.pkl.gz', '.txt', '.png', '.jpg', '.tif', '.ocvmb','.csv']:
            if file_name.endswith(p):
                file_name = file_name.replace(p,'')

        # get now new file name
        true_file_name = os.path.join(folder, file_name + pattern)
        all_files.append(true_file_name)
        labels.append(class_id)

    return all_files, labels

def load_graph(frozen_graph_filename):
    """
    load a pb graph, this file is from pix_lab/util/util.py of the ARU-Net
    """
    import tensorflow as tf
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="",
            op_dict=None,
            producer_op_list=None
        )
    return graph


def loadpkl(fname, outdir=None):
    if not fname.endswith('pkl.gz'):
        fname += '.pkl.gz'
    if outdir:
        assert(os.path.exists(outdir))
        fname = os.path.join(outdir, fname)
    if not os.path.exists(fname):
        print('WARNING: file {} doesnt exist'.format(fname))
        return None

    if sys.version_info >= (3, 0):
 #       try:
        with gzip.open(fname, 'rb') as f:
                #ret = cPickle.load(f, encoding='latin1')
            ret = pickle.load(f)
#        except:
#            print('ERROR: cannot load', fname
            #raise
    else:
        with gzip.open(fname, 'rb') as f:
            ret = cPickle.load(f)
    return ret

def savepkl(fname, what, outdir=None):
    if not fname.endswith('pkl.gz'):
        fname += '.pkl.gz'
    if outdir:
        assert(os.path.exists(outdir))
        fname = os.path.join(outdir, fname)

    if sys.version_info >= (3, 0):
        with gzip.open(fname, 'wb') as fOut:
            pickle.dump(what, fOut, -1)
    else:
        with gzip.open(fname, 'wb') as fOut:
            cPickle.dump(what, fOut, -1)

def adjust_lr_exp(optimizer, base_lr, ep, total_ep, start_decay_at_ep):
    """Decay exponentially in the later phase of training. All parameters in the
    optimizer share the same learning rate.

    Args:
      optimizer: a pytorch `Optimizer` object
      base_lr: starting learning rate
      ep: current epoch, ep >= 1
      total_ep: total number of epochs to train
      start_decay_at_ep: start decaying at the BEGINNING of this epoch

    Example:
      base_lr = 2e-4
      total_ep = 300
      start_decay_at_ep = 201
      It means the learning rate starts at 2e-4 and begins decaying after 200
      epochs. And training stops after 300 epochs.

    NOTE:
      It is meant to be called at the BEGINNING of an epoch.
    """
    assert ep >= 0, "Current epoch number should be >= 0"

    if ep < start_decay_at_ep:
        return

    for g in optimizer.param_groups:
        g['lr'] = (base_lr * (0.001 ** (float(ep + 1 - start_decay_at_ep)
                                        / (total_ep + 1 - start_decay_at_ep))))
    print('=====> lr adjusted to {:.10f}'.format(g['lr']).rstrip('0'))


