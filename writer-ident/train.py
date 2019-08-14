import argparse
import os

import config
import triplet_net
from data import WriterData, ICDAR2013
from encoder import ResNet50Encoder
from triplet_selectors import HardestNegativeTripletSelector


def train():
    parser = argparse.ArgumentParser(description='Train the triplet network.')
    parser.add_argument('tasks', nargs='+', type=str)
    parser.add_argument('--dataset', choices=['historical-wi', 'icdar2013'], action='store',
                        type=str, default='historical-wi')
    parser.add_argument('--epochs', action='store', type=int, default=config.DEFAULT_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=config.TRAIN_BATCH_SIZE)
    parser.add_argument('--guaranteed_triplets', type=int, default=config.GUARANTEED_TRIPLETS)
    parser.add_argument('--context', choices=['default', 'cluster', 'gcloud'], default='default')
    parser.add_argument('--lambda', dest='gmp_lambda', type=float, default=config.DEFAULT_LAMBDA)
    parser.add_argument('--lambda-start', dest='lambda_start',
                        type=int, default=config.DEFAULT_LAMBDA_START)
    parser.add_argument('--lambda-end', dest='lambda_end',
                        type=int, default=config.DEFAULT_LAMBDA_END)
    parser.add_argument('--lambda-multiplier', dest='lambda_multiplier',
                        type=float, default=config.DEFAULT_LAMBDA_MULTIPLIER)
    parser.add_argument('--margin', type=float, default=config.DEFAULT_MARGIN)
    parser.add_argument('--lr', type=float, default=config.DEFAULT_LR)
    parser.add_argument('--lr-feature-extractor', type=float, default=config.DEFAULT_LR)
    parser.add_argument('--num-classes', dest='num_classes',
                        type=int, default=config.DEFAULT_NUM_CLASSES)
    parser.add_argument('--samples-per-class', dest='num_samples',
                        type=int, default=config.DEFAULT_SAMPLES_PER_CLASS)
    parser.add_argument('--optimizer', type=str, default=config.OPTIMIZER)
    parser.add_argument('--lr-start', dest='lr_start', type=float, default=config.DEFAULT_LR_START)
    parser.add_argument('--lr-end', dest='lr_end', type=float, default=config.DEFAULT_LR_END)
    parser.add_argument('--lr-multiplier', dest='lr_multiplier', type=float, default=config.DEFAULT_LR_MULTIPLIER)
    parser.add_argument('--scheduler-step', dest='scheduler_step', type=int, default=config.SCHEDULER_STEP)
    parser.add_argument('--lr-decay', dest='lr_decay', type=float, default=config.LR_DECAY)
    parser.add_argument('--weight-decay', dest='weight_decay', type=float, default=config.WEIGHT_DECAY)
    parser.add_argument('--threshold-features', dest='threshold_features', type=int)
    parser.add_argument('--canny-sigma', dest='canny_sigma', type=int)

    parser.add_argument('--color', dest='color', action='store_true')
    parser.add_argument('--binarized', dest='color', action='store_false')
    parser.set_defaults(color=True)
    parser.add_argument('--use-fc', dest='use_fc', action='store_true')
    parser.set_defaults(use_fc=False)
    parser.add_argument('--use-patches', dest='use_patches', action='store_true')
    parser.set_defaults(use_patches=False)
    parser.add_argument('--trainval', action='store_true')
    parser.set_defaults(trainval=False)
    parser.add_argument('--lr-scheduler', dest='scheduler', choices=['exp', 'step', 'multi-step'],
                        type=str, action='store', default='exp')
    parser.add_argument('--start-exp-decay', dest='start_exp_decay', type=int, action='store', default=151)
    parser.add_argument('--pooling', choices=['avg', 'gem', 'gmp', 'conv', 'max', 'lse', 'gmp-fix', 'mixed-pool'],
                        action='store', default='avg')
    parser.add_argument('--conv-pool-dim', type=int, action='store')
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--val-interval', type=int, default=5)
    parser.add_argument('--experiment-dir', type=str, action='store', default='train')
    parser.add_argument('--lse-r', type=float, action='store', default=10,
                        help="Initial value of the (learned) log-sum-exp pool parameter.")
    parser.add_argument('--pool-lr-multiplier', type=int, default=config.POOL_LR_MULTIPLIER)
    parser.add_argument('--normalize', choices=['l2', 'none'], default='l2',
                        help='apply an l2 normalization')

    args = parser.parse_args()
    config.CONTEXT = args.context
    config.COLOR = args.color
    config.USE_PATCHES = args.use_patches
    config.LR_SCHEDULER = args.scheduler
    config.START_EXP_DECAY = args.start_exp_decay

    data = None
    if args.dataset == 'historical-wi':
        data_dir = config.WRITER_DATA_DIR[args.context]
        data = WriterData(data_dir, args.color, args.trainval)
    elif args.dataset == 'icdar2013':
        data_dir = config.ICDAR2013_DIR[args.context]
        data = ICDAR2013(data_dir)
    print("train on: ", data.train_data)

    if args.canny_sigma:
        config.CANNY_SIGMA = args.canny_sigma

    if args.threshold_features:
        config.THRESHOLD_FEATURES = args.threshold_features
    if args.experiment_dir:
        config.EXPERIMENT_DIR = os.path.join(config.EXPERIMENT_DIR, args.experiment_dir)

    encoder = ResNet50Encoder(pool=args.pooling, gmp_lambda=args.gmp_lambda, conv_dim=args.conv_pool_dim,
                              lse_r=args.lse_r, normalize=args.normalize)

    rankings_output = config.RANKINGS_OUTPUT_DIR[args.context]

    config.VAL_INTERVAL = args.val_interval
    config.POOL_LR_MULTIPLIER = args.pool_lr_multiplier

    if 'train' in args.tasks:
        trainer = triplet_net.Trainer(data, rankings_output, args.optimizer, args.scheduler,
                                      args.num_classes, args.num_samples, encoder)
        trainer.train_net(args.epochs, args.lr, HardestNegativeTripletSelector(args.margin),
                          args.gmp_lambda, args.margin, scheduler_step=args.scheduler_step,
                          lr_decay=args.lr_decay, weight_decay=args.weight_decay, log_interval=args.log_interval)

    if 'lr-and-lambda' in args.tasks:
        trainer = triplet_net.Trainer(data, rankings_output, args.optimizer, args.scheduler,
                                      args.num_classes, args.num_samples, encoder)

        trainer.find_lr_and_lambda(args.lr_start, args.lr_end, args.lr_multiplier,
                                   args.epochs, args.margin, args.scheduler_step, args.lr_decay,
                                   args.weight_decay, args.lambda_start, args.lambda_end,
                                   args.lambda_multiplier)

    if 'find-lambda' in args.tasks:
        print("train lambda")
        trainer = triplet_net.Trainer(data, rankings_output, args.optimizer, args.scheduler,
                                      args.num_classes, args.num_samples, encoder)
        trainer.find_lambda(args.lambda_start, args.lambda_end, args.lambda_multiplier, args.margin,
                            args.lr, args.epochs, args.scheduler_step, args.lr_decay,
                            args.weight_decay)

    if any(e in args.tasks for e in ('find-lr', 'find_lr')):
        trainer = triplet_net.Trainer(data, rankings_output, args.optimizer, args.scheduler,
                                      args.num_classes, args.num_samples, encoder)
        trainer.find_lr(args.lr_start, args.lr_end, args.lr_multiplier, args.gmp_lambda,
                        args.epochs, args.margin, scheduler_step=args.scheduler_step,
                        lr_decay=args.lr_decay, weight_decay=args.weight_decay)


if __name__ == '__main__':
    train()
