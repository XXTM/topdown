"""You Only Look Once Object Detection v3"""
# pylint: disable=arguments-differ
from __future__ import absolute_import
from __future__ import division

import os, numpy as np
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn
from gluoncv.model_zoo.yolo.yolo3 import get_yolov3


__all__ = ['fyolo_vgg_voc', 'fyolo_darknet_voc', 'fyolo_darknet53_voc']

def fyolo_darknet53_voc(version='simple', full_stages=False, pretrained_base=True,
                        pretrained=False, num_sync_bn_devices=-1, **kwargs):
    """FYOLO of darknet53 on VOC dataset, supporting full-stages multiscale prediction

    Parameters
    ----------
    version : str
        The darknet version for detecting low-spatial-frequency signals.
    full_stages : boolean
        Set True to use 3-stages for multiscale detection, or False to use the top features only.
    pretrained_base : boolean
        Whether fetch and load pretrained weights for base network.
    pretrained : boolean
        Whether fetch and load pretrained weights for the entire network.
    num_sync_bn_devices : int
        Number of devices for training. If `num_sync_bn_devices < 2`, SyncBatchNorm is disabled.

    Returns
    -------
    mxnet.gluon.HybridBlock
        Fully hybrid yolo3 network.

    """
    from gluoncv.data import VOCDetection
    from .darknet import get_darknet_lsf
    pretrained_base = False if pretrained else pretrained_base
    base_net = get_darknet_lsf(darknet_version=version, num_layers=52, pretrained=pretrained_base,
                               num_sync_bn_devices=num_sync_bn_devices, **kwargs)

    stages = [base_net.features[:15], base_net.features[15:24], base_net.features[24:]]
    anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    strides = [8, 16, 32]
    filters = [512, 256, 128]
    classes = VOCDetection.CLASSES
    if not full_stages:
        stages = [base_net.features[24:]]
        anchors = [anchors[0] + anchors[1] + anchors[2]]
        strides = [32]
        filters = [512]

    return get_yolov3(
        'darknet53', stages, filters, anchors, strides, classes, 'voc',
        pretrained=pretrained, num_sync_bn_devices=num_sync_bn_devices, **kwargs)

def fyolo_darknet_voc(version='simple', num_layers=52, pretrained_base=True,
                      pretrained=False, num_sync_bn_devices=-1, **kwargs):
    """FYOLO of darknet on VOC dataset, supporting single-scale prediction only

    Parameters
    ----------
    version : str
        The darknet version for detecting low-spatial-frequency signals.
    num_layers : int
        Keep the first num_layers as the feature extractor. 26, 43, 52
    pretrained_base : boolean
        Whether fetch and load pretrained weights for base network.
    pretrained : boolean
        Whether fetch and load pretrained weights for the entire network.
    num_sync_bn_devices : int
        Number of devices for training. If `num_sync_bn_devices < 2`, SyncBatchNorm is disabled.

    Returns
    -------
    mxnet.gluon.HybridBlock
        Fully hybrid yolo3 network.

    """
    from gluoncv.data import VOCDetection
    from .darknet import get_darknet_lsf
    pretrained_base = False if pretrained else pretrained_base
    base_net = get_darknet_lsf(darknet_version=version, num_layers=num_layers, pretrained=pretrained_base,
                               num_sync_bn_devices=num_sync_bn_devices, **kwargs)

    classes = VOCDetection.CLASSES
    # TODO @ xyutao: Implemenet darknet-based single-scale yolo.

def fyolo_vgg_voc(backbone="vgg16", num_layers=13, pretrained_base=True,
                  pretrained=False, num_sync_bn_devices=-1, **kwargs):
    """FYOLO of VGG on VOC dataset

    Parameters
    ----------
    backbone : str
        Use the imagenet pretrained backbone ("vgg11", "vgg13" or "vgg16") for initialization.
    num_layers : int
        Keep the first num_layers of pretrained darknet to build an fnet.
    pretrained_base : boolean
        Whether fetch and load pretrained weights for base network.
    pretrained : boolean
        Whether fetch and load pretrained weights for the entire network.
    num_sync_bn_devices : int
        Number of devices for training. If `num_sync_bn_devices < 2`, SyncBatchNorm is disabled.

    Returns
    -------
    mxnet.gluon.HybridBlock
        Fully hybrid yolo3 network.

    """
    from gluoncv.data import VOCDetection
    from .vgg import get_vgg_lsf
    pretrained_base = False if pretrained else pretrained_base
    base_net = get_vgg_lsf(backbone=backbone, keep_layers=num_layers, pretrained=pretrained_base,
                           num_sync_bn_devices=num_sync_bn_devices, **kwargs)

    classes = VOCDetection.CLASSES
    # TODO @ xyutao: Implemenet vgg-based single-scale yolo.
