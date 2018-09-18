from __future__ import division
__all__ = ['VGGLSF', 'get_vgg_lsf', 'vgg_simple']

from mxnet.context import cpu
from mxnet.initializer import Xavier
from mxnet import gluon
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn

def _conv2d(filter, kernel, dilation, padding, stride):
    """A common conv-bn-leakyrelu cell"""
    cell = nn.HybridSequential(prefix='')
    cell.add(nn.Conv2D(filter, kernel_size=kernel, dilation=dilation,
                       strides=stride, padding=padding,
                       weight_initializer=Xavier(rnd_type='gaussian',
                                                 factor_type='out',
                                                 magnitude=2),
                       bias_initializer='zeros'))
    cell.add(nn.Activation('relu'))
    return cell

def _conv2d_bn(filter, kernel, dilation, padding, stride, num_sync_bn_devices=-1):
    """A common conv-bn-leakyrelu cell"""
    cell = nn.HybridSequential(prefix='')
    cell.add(nn.Conv2D(filter, kernel_size=kernel, dilation=dilation,
                       strides=stride, padding=padding,
                       weight_initializer=Xavier(rnd_type='gaussian',
                                                 factor_type='out',
                                                 magnitude=2),
                       bias_initializer='zeros'))
    if num_sync_bn_devices < 1:
        cell.add(nn.BatchNorm(epsilon=1e-5, momentum=0.9))
    else:
        cell.add(gluon.contrib.nn.SyncBatchNorm(
            epsilon=1e-5, momentum=0.9, num_devices=num_sync_bn_devices))
    cell.add(nn.Activation('relu'))
    return cell


class VGGLSF(HybridBlock):
    r"""VGG Low-spatial-frequency model with dilated convolutions.

    Parameters
    ----------
    layers : iterable
        Description of parameter `layers`.
    channels : iterable
        Description of parameter `channels`.
    dilations : iterable
        Description of parameter `dilations`.
    strides : iterable
        Description of parameter `strides`.
    batch_norm : bool, default False
        Use batch normalization.
    """
    def __init__(self, layers, filters, dilations, strides, batch_norm=False, **kwargs):
        super(VGGLSF, self).__init__(prefix='vgg', **kwargs)
        assert len(layers) == len(filters), (
            "len(filters) should equal to len(layers), given {} vs {}".format(
                len(filters), len(layers)))
        assert len(layers) == len(dilations), (
            "len(dilations) should equal to len(layers), given {} vs {}".format(
                len(dilations), len(layers)))
        assert len(layers) == len(strides), (
            "len(strides) should equal to len(layers), given {} vs {}".format(
                len(strides), len(layers)))
        self.stages = len(layers)
        with self.name_scope():
            self.features = self._make_features(layers, filters, dilations, strides, batch_norm)

    def _make_features(self, layers, filters, dilations, strides, batch_norm):
        featurizer = nn.HybridSequential(prefix='')
        conv2d = _conv2d_bn if batch_norm else _conv2d
        count = 0
        for layer, filter, dilation, stride in zip(layers, filters, dilations, strides):
            d = dilation if isinstance(dilation, tuple) else [dilation] * layer
            s = stride if isinstance(stride, tuple) else [stride] * layer
            for i in range(layer):
                featurizer.add(conv2d(filter, 3, d[i], d[i], s[i]))
            # ignore the last pooling layer
            if count == self.stages - 1:
                break
            count += 1
            featurizer.add(nn.MaxPool2D(strides=2))
        return featurizer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        return x


# Specification
vgg_spec = {
    # 'preset': { depth1: ([layers, filters, dilations, strides]) }
    'vgg11': {
        6: ([1, 1, 2, 2], [64, 128, 256, 512], [6, 1, 1, 2], [3, 1, 1, 1]),
        8: ([1, 1, 2, 2, 2], [64, 128, 256, 512, 512], [6, 1, 1, 1, 2], [3, 1, 1, 1, 1])
    },
    'vgg13': {
        8: ([2, 2, 2, 2], [64, 128, 256, 512], [(4, 2), 1, 1, 2], [(3, 1), 1, 1, 1]),
        10: ([2, 2, 2, 2, 2], [64, 128, 256, 512, 512], [(4, 2), 1, 1, 1, 2], [(3, 1), 1, 1, 1, 1]),
    },
    'vgg16': {
        10: ([2, 2, 3, 3], [64, 128, 256, 512], [(4, 2), 1, 1, 2], [(3, 1), 1, 1, 1]),
        13: ([2, 2, 3, 3, 3], [64, 128, 256, 512, 512], [(4, 2), 1, 1, 1, 2], [(3, 1), 1, 1, 1, 1])
    }
}


# Constructors
def get_vgg_lsf(backbone, keep_layers, pretrained=True, ctx=cpu(),
            root='/data/models/gluon/cv', **kwargs):
    r"""VGG model from the `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556>`_ paper.

    Parameters
    ----------
    backbone : str
        'vgg11', 'vgg13' or 'vgg16' backbone.
    keep_layers: int
        keep the first xx layers.
    pretrained : bool, default True
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    """
    layers, filters, dilations, strides = vgg_spec[backbone][keep_layers]
    net = VGGLSF(layers, filters, dilations, strides, **kwargs)
    if pretrained:
        from gluoncv.model_store import get_model_file
        batch_norm_suffix = '_bn' if kwargs.get('batch_norm') else ''
        model_file = get_model_file('%s%s'%(backbone, batch_norm_suffix), root=root)
        net.load_parameters(model_file, allow_missing=True, ignore_extra=True, ctx=ctx)
    return net

def vgg_simple(**kwargs):
    return get_vgg_lsf('vgg11', 8, pretrained=True, **kwargs)