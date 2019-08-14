import argparse
import os
import torch
import torchvision
from torchvision import transforms
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

import config
from data import WriterData
import evaluation
from data_transforms import PatchCrop
from encoder import ResNet50Encoder
from datetime import datetime


def descriptors_per_document(model, dataloader):
    """
    Computes a global descriptor for each document in the given dataset.

    Args:

        model: the encoding network to be applied.
        dataloader: The dataloader of the dataset,
                    should retrieve every document in an epoch exactly once.

    Returns:
        encodings: Tensor consisting of the encodings for each document
        targets: target class of the samples
    """
    encodings = []
    writers = []
    i = 0
    for data, target in dataloader:
        if torch.cuda.is_available():
            data = data.cuda()
        # encoding = model(data)
        encoding = model.extract_features(data)
        encoding = encoding.detach().cpu().numpy()
        encodings.append(encoding)
        if i % 100 == 0:
            print('{} / {}'.format(i, len(dataloader)))
        i += 1
        writers.append(target.detach().cpu().numpy())

    encodings = np.vstack(encodings)
    writers = np.hstack(writers)
    return encodings, writers


def descriptors_for_patched_documents(model, dataloader, patch_size, channels):
    print(type(model))
    # basically as above but with averaging
    averaged_encodings = []
    targets = []
    normalize = torchvision.transforms.Normalize(config.MEAN_WRITER, config.STD_WRITER)
    i = 0
    length = len(dataloader)
    for patches, target, in dataloader:
        # print(patches)
        if config.COLOR:
            patches = torch.stack([normalize(patch.view(channels, patch_size, patch_size)) for patch in patches])
        else:
            patches = torch.stack([patch.view(channels, patch_size, patch_size) for patch in patches])
        # print(patches)
        patches = patches.view(-1, channels, patch_size, patch_size)

        if torch.cuda.is_available():
            patches = patches.cuda()
        # encoding = model(patches)
        encoding = model.extract_features(patches)
        encoding = encoding.mean(dim=0)
        encoding = encoding.detach().cpu().numpy()
        averaged_encodings.append(encoding)
        if i % 50 == 0:
            print('apply model: {} / {} samples'.format(i, length))

        i += 1
        targets.append(target.detach().cpu().numpy())

    # print("before stack ", averaged_encodings)
    averaged_encodings = np.vstack(averaged_encodings)
    # print("\nafter stack ", averaged_encodings)

    targets = np.hstack(targets)
    return averaged_encodings, targets


def retrieve_most_similar_documents(encodings, writers, query_document, n=5):
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
    similarities = euclidean_distances(query_document.reshape(1, -1), encodings)
    similarities = similarities.reshape(-1)
    df = pd.DataFrame({'writer': writers, 'similarity': similarities})
    df = df.sort_values('similarity', ascending=True)
    df.to_csv("similarities {}".format(query_document[0]))
    return df


def compute_similarity_rankings(encodings, targets):
    # distance matrix: i-th row corresponds to ith sample
    dist_matrix = euclidean_distances(encodings, squared=True)
    df = pd.DataFrame()
    for i in range(dist_matrix.shape[0]):
        query_name = 'q{}_dist'.format(i)
        query_targets = 'q{}_class'.format(i)
        df[query_name] = dist_matrix[i]
        df[query_targets] = targets
        df[query_targets] = df[query_targets].astype('int32')
    # sort associated columns in the data frame
    for i in range(0, len(df.columns), 2):
        s = df[df.columns[i: i + 2]]
        s = s.sort_values(df.columns[i])
        df[df.columns[i: i + 2]] = s.values
    return df


def evaluate(model, uses_patches: bool, patch_size=400, channels=3, color=False):
    print("eval")
    model.eval()
    data_dir = config.WRITER_DATA_DIR[config.CONTEXT]
    data = WriterData(data_dir, color)
    if uses_patches:
        trans = torchvision.transforms.Compose([
            PatchCrop(patch_size),
            torchvision.transforms.Lambda(
                lambda crops: [torchvision.transforms.ToTensor()(crop) for crop in crops]),

        ])
        # dataloader = data.get_test_data_loader(transform=trans)
        dataloader = data.get_test_data_loader(trans)
        print("comp descriptors")
        encodings, targets = descriptors_for_patched_documents(model, dataloader, patch_size, channels)
    else:
        dataloader = data.get_competition_data_loader()
        encodings, targets = descriptors_per_document(model, dataloader)
        print("envodings length", len(encodings), " targets length: ", len(targets))
    df = compute_similarity_rankings(encodings, targets)
    df.to_csv('rankings_{}.csv'.format(datetime.now().strftime('%m-%d_%H:%M')))
    mAP = evaluation.mean_avg_precision(df, relevant_docs=4, starting_col=1)
    top1 = [df[df.columns[i]][0] == df[df.columns[i]][1] for i in range(1, len(df.columns), 2)]
    top1 = sum(top1) / len(top1)
    print("mAP: {}, Top1: {}".format(mAP, top1))
    return mAP, top1


def main():
    parser = argparse.ArgumentParser(description="Apply a given model")
    parser.add_argument('model_path', nargs='?', type=str)
    parser.add_argument('--dataset', choices=['historical-wi', ], action='store',
                        type=str, default='historical-wi')
    parser.add_argument('--use-patches', dest='use_patches', action='store_true')
    parser.set_defaults(use_patches=False)
    parser.add_argument('--use-fc', dest='use_fc', action='store_true')
    parser.set_defaults(use_fc=False)
    parser.add_argument('--binarized', dest='color', action='store_false')
    parser.set_defaults(color=True)
    parser.add_argument('--pool', type=str, choices=['avg', 'gem', 'gmp', 'conv', 'max', 'lse'])

    args = parser.parse_args()

    encoder = ResNet50Encoder(pool=args.pool)

    if os.path.isdir(args.model_path):
        # Path to directory. Apply all models found there.
        files = [f for f in os.listdir(args.model_path)
                 if os.path.join(args.model_path, f)]
        for f in files:
            print("load: ", f)
            encoder.load_state_dict(torch.load(os.path.join(args.model_path, f)))
            encoder = encoder.cuda()
            evaluate(encoder, uses_patches=args.use_patches, color=args.color)
    else:
        # Path to a  single model.
        print(args.model_path)
        # print(torch.load(args.model_path))
        encoder.load_state_dict(state_dict=torch.load(args.model_path))
        print(encoder.pool.lamb)
        encoder = encoder.cuda()
        evaluate(encoder, uses_patches=args.use_patches, color=args.color)


if __name__ == '__main__':
    main()