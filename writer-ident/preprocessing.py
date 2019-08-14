import argparse

import config
import utils


def main():
    parser = argparse.ArgumentParser("Preprocess the data")
    parser.add_argument('task', nargs='?', type=str)
    parser.add_argument('--path', '-p', dest='path', action='store', type=str)
    parser.add_argument('--patch-size', dest='patch_size', action='store', type=int,
                        default=[], nargs='+')
    parser.add_argument('--canny-sigma', dest='canny_sigma', action='store', type=float)
    parser.add_argument('--threshold', type=int)
    parser.add_argument('--color', dest='color', action='store_true')
    parser.add_argument('--binarized', dest='color', action='store_false')
    parser.add_argument('--val-size', dest='val_size', type=float)
    parser.set_defaults(color=True)
    parser.add_argument('--patch-stride', dest='patch_stride', type=int, default=256)
    parser.add_argument('--padding', type=int, default=0)

    args = parser.parse_args()

    if args.task == 'patchify':
        # split images into patches
        utils.dataset_to_patches(args.path, args.patch_size,
                                 stride=args.patch_stride,
                                 canny_sigma=args.canny_sigma,
                                 threshold=args.threshold,
                                 color=args.color,
                                 padding=args.padding)
    if args.task == 'split-writer-dirs':
        utils.prepare_files_of_trainingset(args.path)

    if args.task == 'train-val-split':
        utils.train_test_split(args.path, args.val_size)


if __name__ == '__main__':
    main()