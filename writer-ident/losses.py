import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)
        # print(len(triplets))
        # print(triplets)
        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)

        if self.margin:
            losses = F.relu(ap_distances - an_distances + self.margin)
        else:
            # Soft margin uses softplus as a smooth approximation
            losses = F.softplus(ap_distances - an_distances)

        return losses.mean(), len(triplets)


class GMPLoss(nn.Module):

    def __init__(self, margin, alpha=1):
        super(GMPLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha

    def forward(self, f, y):
        f = f.view(f.size(0), f.size(1), -1)
        y = y.view(y.size(0), y.size(1), 1)
        prod = torch.bmm(f.permute(0, 2, 1), y)
        return self.alpha * torch.nn.functional.relu(torch.norm(prod - self.margin, p=2))
