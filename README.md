# Deep Generalized Max Pooling (DGMP)

![DGMP Overview](dgmp.png)

This repository contains the code to reproduce the evaluations
of the following publication:

_V. Christlein, L. Spranger, M. Seuret, A. Nicolaou, P. Král, A. Maier,
"Deep Generalized Max Pooling," in 15th International Conference on Document Analysis and Recognition, Sep. 2019, Sydney, Australia_

## Abstract:
> Global pooling layers are an essential part of Convolutional Neural Networks
> (CNN). They are used to aggregate activations of spatial locations to produce a
> fixed-size vector in several state-of-the-art CNNs. Global average pooling or
> global max pooling are commonly used for converting convolutional features of
> variable size images to a fix-sized embedding. However, both pooling layer
> types are computed spatially independent: each individual activation map is
> pooled and thus activations of different locations are pooled together. In
> contrast, we propose Deep Generalized Max Pooling that balances the
> contribution of all activations of a spatially coherent region by re-weighting
> all descriptors so that the impact of frequent and rare ones is equalized. We
> show that this layer is superior to both average and max pooling on the
> classification of Latin medieval manuscripts (CLAMM’16, CLAMM’17), as well as
> writer identification (Historical-WI’17).

tldr: Use DGMP as global pooling layer to get some more accuracy

## Requirements
Both code-bases are independent.

The clamm code only requires:
 - pytorch >= 1.0
 - sklearn

## Run
Have a look at the _scripts_ folder.

Just looking for the pooling layer? -> see _pooling.py_ in one of the folders.
