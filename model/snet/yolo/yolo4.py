"""You Only Look Once Object Detection v3"""
# pylint: disable=arguments-differ
from __future__ import absolute_import
from __future__ import division

import os
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from gluoncv.model_zoo.yolo.darknet import _conv2d, darknet53
from gluoncv.model_zoo.yolo.yolo3 import YOLOOutputV3, YOLOV3
from gluoncv.model_zoo.yolo.yolo_target import YOLOV3TargetMerger
from gluoncv.loss import YOLOV3Loss

__all__ = ['YOLOV4', 'get_yolov4', 'yolo4_darknet53_voc']

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

class YOLODetectionBlockV4(gluon.HybridBlock):
    """YOLO V3 Detection Block which does the following:

    - add a few conv layers
    - return the output
    - have a branch that do yolo detection.

    Parameters
    ----------
    channel : int
        Number of channels for 1x1 conv. 3x3 Conv will have 2*channel.
    num_sync_bn_devices : int, default is -1
        Number of devices for training. If `num_sync_bn_devices < 2`, SyncBatchNorm is disabled.

    """
    def __init__(self, channel, dilations, num_sync_bn_devices=-1, **kwargs):
        super(YOLODetectionBlockV4, self).__init__(**kwargs)
        assert channel % 2 == 0, "channel {} cannot be divided by 2".format(channel)
        with self.name_scope():
            d = dilations
            self.body = nn.HybridSequential(prefix='')
            for i in range(2):
                # 1x1 reduce
                self.body.add(_conv2d(channel, 1, 0, 1, num_sync_bn_devices))
                # 3x3 expand
                self.body.add(_conv2d_dl(channel * 2, 3, d[i], d[i], 1, num_sync_bn_devices))
            self.body.add(_conv2d(channel, 1, 0, 1, num_sync_bn_devices))
            self.tip = _conv2d_dl(channel * 2, 3, d[-1], d[-1], 1, num_sync_bn_devices)

    # pylint: disable=unused-argument
    def hybrid_forward(self, F, x):
        route = self.body(x)
        tip = self.tip(route)
        return route, tip


class YOLOV4(YOLOV3):
    """YOLO V4 detection network.

    Parameters
    ----------
    dilations : iterable
        An embedded List of dilations for the 2 head blocks and 1 tip layer, of each stage.
    stages : mxnet.gluon.HybridBlock
        Staged feature extraction blocks.
        For example, 3 stages and 3 YOLO output layers are used original paper.
    channels : iterable
        Number of conv channels for each appended stage.
        `len(channels)` should match `len(stages)`.
    num_class : int
        Number of foreground objects.
    anchors : iterable
        The anchor setting. `len(anchors)` should match `len(stages)`.
    strides : iterable
        Strides of feature map. `len(strides)` should match `len(stages)`.
    alloc_size : tuple of int, default is (128, 128)
        For advanced users. Define `alloc_size` to generate large enough anchor
        maps, which will later saved in parameters. During inference, we support arbitrary
        input image by cropping corresponding area of the anchor map. This allow us
        to export to symbol so we can run it in c++, Scalar, etc.
    nms_thresh : float, default is 0.45.
        Non-maximum suppression threshold. You can speficy < 0 or > 1 to disable NMS.
    nms_topk : int, default is 400
        Apply NMS to top k detection results, use -1 to disable so that every Detection
         result is used in NMS.
    post_nms : int, default is 100
        Only return top `post_nms` detection results, the rest is discarded. The number is
        based on COCO dataset which has maximum 100 objects per image. You can adjust this
        number if expecting more objects. You can use -1 to return all detections.
    pos_iou_thresh : float, default is 1.0
        IOU threshold for true anchors that match real objects.
        'pos_iou_thresh < 1' is not implemented.
    ignore_iou_thresh : float
        Anchors that has IOU in `range(ignore_iou_thresh, pos_iou_thresh)` don't get
        penalized of objectness score.
    num_sync_bn_devices : int, default is -1
        Number of devices for training. If `num_sync_bn_devices < 2`, SyncBatchNorm is disabled.

    """
    def __init__(self, dilations, stages, channels, anchors, strides, classes, alloc_size=(128, 128),
                 nms_thresh=0.45, nms_topk=400, post_nms=100, pos_iou_thresh=1.0,
                 ignore_iou_thresh=0.7, num_sync_bn_devices=-1, **kwargs):
        super(YOLOV4, self).__init__(stages, channels, anchors, strides, classes, **kwargs)
        self._classes = classes
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms
        self._pos_iou_thresh = pos_iou_thresh
        self._ignore_iou_thresh = ignore_iou_thresh
        if pos_iou_thresh >= 1:
            self._target_generator = YOLOV3TargetMerger(len(classes), ignore_iou_thresh)
        else:
            raise NotImplementedError(
                "pos_iou_thresh({}) < 1.0 is not implemented!".format(pos_iou_thresh))
        self._loss = YOLOV3Loss()
        with self.name_scope():
            self.stages = nn.HybridSequential()
            self.transitions = nn.HybridSequential()
            self.yolo_blocks = nn.HybridSequential()
            self.yolo_outputs = nn.HybridSequential()
            # note that anchors and strides and dilations should be used in reverse order
            for i, stage, channel, anchor, stride, dilation in zip(
                    range(len(stages)), stages, channels, anchors[::-1], strides[::-1], dilations[::-1]):
                self.stages.add(stage)
                block = YOLODetectionBlockV4(channel, dilation, num_sync_bn_devices)
                self.yolo_blocks.add(block)
                output = YOLOOutputV3(i, len(classes), anchor, stride, alloc_size=alloc_size)
                self.yolo_outputs.add(output)
                if i > 0:
                    self.transitions.add(_conv2d(channel, 1, 0, 1, num_sync_bn_devices))


def get_yolov4(name, dilations, stages, filters, anchors, strides, classes,
               dataset, pretrained=False, ctx=mx.cpu(),
               root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    """Get YOLOV3 models.

    Parameters
    ----------
    name : str or None
        Model name, if `None` is used, you must specify `features` to be a `HybridBlock`.
    dilations : iterable
        A List of dilations for the 2 head blocks and 1 tip layer
    stages : iterable of str or `HybridBlock`
        List of network internal output names, in order to specify which layers are
        used for predicting bbox values.
        If `name` is `None`, `features` must be a `HybridBlock` which generate mutliple
        outputs for prediction.
    filters : iterable of float or None
        List of convolution layer channels which is going to be appended to the base
        network feature extractor. If `name` is `None`, this is ignored.
    sizes : iterable fo float
        Sizes of anchor boxes, this should be a list of floats, in incremental order.
        The length of `sizes` must be len(layers) + 1. For example, a two stage SSD
        model can have ``sizes = [30, 60, 90]``, and it converts to `[30, 60]` and
        `[60, 90]` for the two stages, respectively. For more details, please refer
        to original paper.
    ratios : iterable of list
        Aspect ratios of anchors in each output layer. Its length must be equals
        to the number of SSD output layers.
    steps : list of int
        Step size of anchor boxes in each output layer.
    classes : iterable of str
        Names of categories.
    dataset : str
        Name of dataset. This is used to identify model name because models trained on
        differnet datasets are going to be very different.
    pretrained : bool, optional, default is False
        Load pretrained weights.
    pretrained_base : bool, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `Ture`, this has no effect.
    ctx : mxnet.Context
        Context such as mx.cpu(), mx.gpu(0).
    root : str
        Model weights storing path.

    Returns
    -------
    HybridBlock
        A YOLOV3 detection network.
    """
    net = YOLOV4(dilations, stages, filters, anchors, strides, classes, **kwargs)
    if pretrained:
        from gluoncv.model_store import get_model_file
        full_name = '_'.join(('yolo3', name, dataset))
        net.load_params(get_model_file(full_name, root=root), ctx=ctx)
    return net

def yolo4_darknet53_voc(pretrained_base=True, pretrained=False, num_sync_bn_devices=-1, **kwargs):
    """YOLO3 multi-scale with darknet53 base network on VOC dataset.

    Parameters
    ----------
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
    pretrained_base = False if pretrained else pretrained_base
    base_net = darknet53(pretrained=pretrained_base, num_sync_bn_devices=num_sync_bn_devices)
    stages = [base_net.features[:15], base_net.features[15:24], base_net.features[24:]]
    anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    strides = [8, 16, 32]
    classes = VOCDetection.CLASSES
    # Here for adjusting dilations
    dilations = [[1, 1, 1], [1, 2, 2], [2, 2, 4]]
    return get_yolov4(
        'darknet53', dilations, stages, [512, 256, 128], anchors, strides, classes, 'voc',
        pretrained=pretrained, num_sync_bn_devices=num_sync_bn_devices, **kwargs)
