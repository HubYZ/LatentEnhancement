import threading
import torch
import numpy as np
from scipy import ndimage
import torch.nn.functional as F


class MyMetric:
    def __init__(self):
        # P is an attack
        # N is not an attack, is a normal measurement
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0

    def update(self, labels, preds):
        # preds = F.softmax(preds,dim=1)
        # preds = preds[:,1]

        labels_P = labels == 1
        labels_N = labels == 0
        pred_P = preds >= 0.5
        pred_N = preds < 0.5

        num_TP = (labels_P & pred_P).sum()
        num_FP = (labels_N & pred_P).sum()
        num_FN = (labels_P & pred_N).sum()
        num_TN = (labels_N & pred_N).sum()

        self.TP += num_TP
        self.FP += num_FP
        self.FN += num_FN
        self.TN += num_TN

    def get(self):
        prec = self.TP/(self.TP+self.FP)
        recall = self.TP/(self.TP+self.FN)
        # acc = (self.TP+self.TN)/(self.TP+self.TN+self.FP+self.FN)
        F1 = 2*prec*recall/(prec+recall)
        return prec, recall, F1

    def reset(self):
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0


class SegmentationMetric(object):
    """Computes pixAcc and mIoU metric scroes"""

    def __init__(self, channel_axis=1, foreground_class_index=[1], three_d=False):
        self.channel_axis = channel_axis
        self.foreground_class_index = foreground_class_index
        self.three_d = three_d

        self.total_union = 0

        if not self.three_d:
            self.strel = np.zeros([3,3])
            self.strel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        else:
            self.strel = np.zeros([3, 3, 3, 3, 3])
            self.strel[1][1] = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])

    def connencted_components(self, input_np):
        labeled = np.zeros_like(input_np, dtype=np.uint16)
        max_num = 0
        for d1, image in enumerate(input_np):
            labeled_image, max_num_temp = ndimage.label(image, structure=self.strel)
            labeled[d1] = labeled_image
            max_num = np.maximum(max_num, max_num_temp)

        return labeled, np.array(max_num).astype(np.float32)

    def seg_numpy(self, gt, seg):
        gt_labled, _ = self.connencted_components(gt.astype(np.int8))
        seg_labled, _ = self.connencted_components(seg.astype(np.int8))

        all_iou = []
        for gt, seg in zip(gt_labled, seg_labled):
            for this_label in np.unique(gt):
                if this_label == 0:
                    continue
                all_iou.append(0.)
                bw = gt == this_label
                l_area = np.sum(bw).astype(np.float32)
                overlaping_inds = seg[bw]
                for s in np.unique(overlaping_inds):
                    if s == 0:
                        continue
                    intersection = np.sum(overlaping_inds == s).astype(np.float32)
                    overlap = intersection / l_area
                    if overlap > 0.5:
                        s_area = np.sum(seg == s).astype(np.float32)
                        iou = intersection / (l_area + s_area - intersection)
                        all_iou[-1] = iou
        if not len(all_iou):
            return np.nan
        return np.mean(all_iou)


    def calc_seg(self, gt_sequence, output_sequence):
        output_sequence = output_sequence.detach().cpu().numpy()
        gt_sequence     = gt_sequence.detach().cpu().numpy().astype(np.long)

        output_classes = np.argmax(output_sequence, axis=self.channel_axis)

        gt_valid = gt_sequence > -1
        gt_sequence = gt_sequence * gt_valid

        # here, we only measure the cell without counting cell border
        gt_fg = gt_sequence == self.foreground_class_index[0]
        output_foreground = output_classes == self.foreground_class_index[0]

        # gt_fg = np.zeros_like(gt_sequence).astype(np.bool)
        # output_foreground = np.zeros_like(output_classes).astype(np.bool)
        # for class_index in self.foreground_class_index:
        #     gt_fg = gt_fg | (gt_sequence == class_index)
        #     output_foreground = output_foreground | (output_classes == class_index)

        seg_measure_value = self.seg_numpy(gt_fg,output_foreground)
        return seg_measure_value


class SegmentationMetric_old(object):
    """Computes pixAcc and mIoU metric scroes"""
    def __init__(self, nclass):
        self.nclass = nclass
        self.lock = threading.Lock()
        self.reset()

    def update(self, labels, preds):
        def evaluate_worker(self, label, pred):
            correct, labeled = batch_pix_accuracy(pred, label)
            inter, union = batch_intersection_union(pred, label, self.nclass)
            with self.lock:
                self.total_correct += correct
                self.total_label += labeled
                self.total_inter += inter
                self.total_union += union
            return

        if isinstance(preds, torch.Tensor):
            evaluate_worker(self, labels, preds)
        elif isinstance(preds, (list, tuple)):
            threads = [threading.Thread(target=evaluate_worker,
                                        args=(self, label, pred),)
                       for (label, pred) in zip(labels, preds)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            raise NotImplemented

    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU
 
    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0
        return

def batch_pix_accuracy(output, target):
    """Batch Pixel Accuracy
    Args:
        predict: input 4D tensor
        target: label 3D tensor
    """
    predict = torch.max(output, 1)[1]

    # label: 0, 1, ..., nclass - 1
    # Note: 0 is background
    predict = predict.cpu().numpy().astype('int64') + 1
    target = target.cpu().numpy().astype('int64') + 1

    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((predict == target)*(target > 0))
    assert pixel_correct <= pixel_labeled, \
        "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled

def batch_intersection_union(output, target, nclass):
    """Batch Intersection of Union
    Args:
        predict: input 4D tensor
        target: label 3D tensor
        nclass: number of categories (int)
    """
    predict = torch.max(output, 1)[1]
    mini = 1
    maxi = nclass-1
    nbins = nclass-1

    # label is: 0, 1, 2, ..., nclass-1
    # Note: 0 is background
    predict = predict.cpu().numpy().astype('int64') + 1
    target = target.cpu().numpy().astype('int64') + 1

    predict = predict * (target > 0).astype(predict.dtype)
    intersection = predict * (predict == target)

    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), \
        "Intersection area should be smaller than Union area"
    return area_inter, area_union

# ref https://github.com/CSAILVision/sceneparsing/blob/master/evaluationCode/utils_eval.py
def pixel_accuracy(im_pred, im_lab):
    im_pred = np.asarray(im_pred)
    im_lab = np.asarray(im_lab)

    # Remove classes from unlabeled pixels in seg_mask image.
    # We should not penalize detections in unlabeled portions of the image.
    pixel_labeled = np.sum(im_lab > 0)
    pixel_correct = np.sum((im_pred == im_lab) * (im_lab > 0))

    return pixel_correct, pixel_labeled

def intersection_and_union(im_pred, im_lab, num_class):
    im_pred = np.asarray(im_pred)
    im_lab = np.asarray(im_lab)
    # Remove classes from unlabeled pixels in seg_mask image.
    im_pred = im_pred * (im_lab > 0)
    # Compute area intersection:
    intersection = im_pred * (im_pred == im_lab)
    area_inter, _ = np.histogram(intersection, bins=num_class-1,
                                        range=(1, num_class - 1))
    # Compute area union: 
    area_pred, _ = np.histogram(im_pred, bins=num_class-1,
                                range=(1, num_class - 1))
    area_lab, _ = np.histogram(im_lab, bins=num_class-1,
                               range=(1, num_class - 1))
    area_union = area_pred + area_lab - area_inter
    return area_inter, area_union

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc

def rel_abs_vol_diff(y_true, y_pred):

    return np.abs( (y_pred.sum()/y_true.sum() - 1)*100)

def get_boundary(data, img_dim=2, shift = -1):
    data  = data>0
    edge = np.zeros_like(data)
    for nn in range(img_dim):
        edge += ~(data ^ np.roll(~data,shift=shift,axis=nn))
    return edge.astype(int)

def numpy_dice(y_true, y_pred, axis=None, smooth=1.0):
    intersection = y_true*y_pred
    return ( 2. * intersection.sum(axis=axis) +smooth )/ (np.sum(y_true, axis=axis) + np.sum(y_pred, axis=axis) + smooth )

def dice_coefficient(input, target, smooth=1.0):
    assert smooth > 0, 'Smooth must be greater than 0.'

    probs = F.softmax(input, dim=1)

    encoded_target = probs.detach() * 0
    encoded_target.scatter_(1, target.unsqueeze(1), 1)
    encoded_target = encoded_target.float()

    num = probs * encoded_target   # b, c, h, w -- p*g
    num = torch.sum(num, dim=3)    # b, c, h
    num = torch.sum(num, dim=2)    # b, c

    den1 = probs * probs           # b, c, h, w -- p^2
    den1 = torch.sum(den1, dim=3)  # b, c, h
    den1 = torch.sum(den1, dim=2)  # b, c

    den2 = encoded_target * encoded_target  # b, c, h, w -- g^2
    den2 = torch.sum(den2, dim=3)  # b, c, h
    den2 = torch.sum(den2, dim=2)  # b, c

    dice = (2 * num + smooth) / (den1 + den2 + smooth) # b, c

    return dice.mean().mean()


