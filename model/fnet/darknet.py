"""Darknet as YOLO backbone network."""
# pylint: disable=arguments-differ
from __future__ import absolute_import

import os
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn

from gluoncv.model_zoo.yolo.darknet import _conv2d

__all__ = ['DarknetLSFSimple', 'get_darknet_lsf', 'darknet_simple']

def _conv2d_dl(channel, kernel, dilation, padding, stride, num_sync_bn_devices=-1):
    """A common conv-bn-leakyrelu cell"""
    cell = nn.HybridSequential(prefix='')
    cell.add(nn.Conv2D(channel, kernel_size=kernel, dilation=dilation,
                       strides=stride, padding=padding, use_bias=False))
    if num_sync_bn_devices < 1:
        cell.add(nn.BatchNorm(epsilon=1e-5, momentum=0.9))
    else:
        cell.add(gluon.contrib.nn.SyncBatchNorm(
            epsilon=1e-5, momentum=0.9, num_devices=num_sync_bn_devices))
    cell.add(nn.LeakyReLU(0.1))
    return cell


class DarknetDLBlock(gluon.HybridBlock):
    """Darknet Dilated Block. Which is a 1x1 reduce conv followed by 3x3 dilated conv.
    The 3x3 dilated conv has dilation=2, padding=2 and stride=1

    Parameters
    ----------
    channel : int
        Convolution channels for 1x1 conv.
    num_sync_bn_devices : int, default is -1
        Number of devices for training. If `num_sync_bn_devices < 2`, SyncBatchNorm is disabled.

    """
    def __init__(self, channel, dilation, num_sync_bn_devices=-1, **kwargs):
        super(DarknetDLBlock, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        # 1x1 reduce
        self.body.add(_conv2d(channel, 1, 0, 1, num_sync_bn_devices))
        # 3x3 conv expand
        self.body.add(_conv2d_dl(channel * 2, 3, dilation, dilation, 1, num_sync_bn_devices))

    # pylint: disable=unused-argument
    def hybrid_forward(self, F, x, *args):
        residual = x
        x = self.body(x)
        return x + residual


class DarknetLSFBlock(gluon.HybridBlock):
    """Darknet Low-Spatial-Frequency Block. Which is a 1x1 reduce conv followed by 3x3 dilated conv.
    The 3x3 dilated conv has dilation=d, padding=d//2 and stride=d
    We use a pooling operation on the residual path to downsample the resolution.

    Parameters
    ----------
    channel : int
        Convolution channels for 1x1 conv.
    num_sync_bn_devices : int, default is -1
        Number of devices for training. If `num_sync_bn_devices < 2`, SyncBatchNorm is disabled.

    """
    def __init__(self, channel, dilation, num_sync_bn_devices=-1, **kwargs):
        super(DarknetLSFBlock, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        # 1x1 reduce
        self.body.add(_conv2d(channel, 1, 0, 1, num_sync_bn_devices))
        # 3x3 conv expand
        self.body.add(_conv2d_dl(channel * 2, 3, dilation, dilation//2, dilation, num_sync_bn_devices))
        # downsample path. compute pool_size
        pool_size = 3 + 2 * (dilation - 1)
        self.downsample = nn.HybridSequential(prefix='')
        self.downsample.add(nn.MaxPool2D(pool_size=pool_size, strides=dilation, padding=dilation))

    # pylint: disable=unused-argument
    def hybrid_forward(self, F, x, *args):
        x = self.body(x)
        residual = self.downsample(x)
        return x + residual


class DarknetLSFSimple(gluon.HybridBlock):
    """Darknet Low-spatial-frequency.

    Parameters
    ----------
    layers : iterable
        Description of parameter `layers`.
    channels : iterable
        Description of parameter `channels`.
    dilations : iterable
        Description of parameter `dilations`.
    num_sync_bn_devices : int, default is -1
        Number of devices for training. If `num_sync_bn_devices < 2`, SyncBatchNorm is disabled.

    Attributes
    ----------
    features : mxnet.gluon.nn.HybridSequential
        Feature extraction layers.
    output : mxnet.gluon.nn.Dense
        A classes(1000)-way Fully-Connected Layer.

    """
    def __init__(self, layers, channels, dilations, num_sync_bn_devices=-1, **kwargs):
        super(DarknetLSFSimple, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1, (
            "len(channels) should equal to len(layers) + 1, given {} vs {}".format(
                len(channels), len(layers)))
        assert len(layers) == len(dilations) - 1, (
            "len(dilations) should equal to len(layers) + 1, given {} vs {}".format(
                len(dilations), len(layers)))
        self._stride = dilations[0] * pow(2, len(layers))
        with self.name_scope():
            self.features = nn.HybridSequential()
            # first 3x3 conv with channels[0], dilation=padding=stride=dilations[0]
            d = dilations[0]
            self.features.add(_conv2d_dl(channels[0], 3, d, d, d, num_sync_bn_devices))
            for nlayer, channel, dilation in zip(layers, channels[1:], dilations[1:]):
                assert channel % 2 == 0, "channel {} cannot be divided by 2".format(channel)
                # add downsample conv with stride=2
                self.features.add(_conv2d(channel, 3, 1, 2, num_sync_bn_devices))
                # add nlayer basic blocks
                for _ in range(nlayer):
                    self.features.add(DarknetDLBlock(channel // 2, dilation, num_sync_bn_devices))
            # try adding downsample conv for the output features
            # self.features.add(_conv2d(channel, 3, 1, 2, num_sync_bn_devices))
            # self._stride *= 2

    @property
    def stride(self):
        return self._stride


    def hybrid_forward(self, F, x):
        return self.features(x)


# default configurations
darknet_versions = {'simple': DarknetLSFSimple}
darknet_spec = {
    'simple': {
        # format preset:([layers, channels, dilations, feat_stride])
        # the first element of channels and dilations are for the first conv-layer
        26: ([1, 2, 8], [32, 64, 128, 256], [3, 1, 1, 2]),
        43: ([1, 2, 8, 8], [32, 64, 128, 256, 512], [3, 1, 1, 1, 2]),
        52: ([1, 2, 8, 8, 4], [32, 64, 128, 256, 512, 1024], [3, 1, 1, 1, 1, 2])}
}

def get_darknet_lsf(darknet_version, num_layers, pretrained=True, ctx=mx.cpu(),
                    root=os.path.join('~/.mxnet/models'), **kwargs):
    """Get darknet by `version` and `num_layers` info.

    Parameters
    ----------
    darknet_version : str
        Darknet version, choices are ['v3'].
    num_layers : int
        Number of layers. 26, 43 or 52.
    pretrained : boolean
        Whether fetch and load pre-trained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    mxnet.gluon.HybridBlock
        Darknet network.

    Examples
    --------
    >>> model = get_darknet_lsf('simple', 53, pretrained=True)
    >>> print(model)

    """
    assert darknet_version in darknet_versions and darknet_version in darknet_spec, (
        "Invalid darknet version: {}. Options are {}".format(
            darknet_version, str(darknet_versions.keys())))
    specs = darknet_spec[darknet_version]
    assert num_layers in specs, (
        "Invalid number of layers: {}. Options are {}".format(num_layers, str(specs.keys())))
    layers, channels, dilations = specs[num_layers]
    darknet_class = darknet_versions[darknet_version]
    net = darknet_class(layers, channels, dilations, **kwargs)
    if pretrained:
        from gluoncv.model_zoo.model_store import get_model_file
        model_file = get_model_file('darknet53', root=root)
        net.load_parameters(model_file, ctx=ctx, allow_missing=True, ignore_extra=True)
    return net

def darknet_simple(**kwargs):
    return get_darknet_lsf('simple', 26, pretrained=True, **kwargs)
