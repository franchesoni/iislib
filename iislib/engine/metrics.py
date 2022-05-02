# from OpenMMLab
# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def get_dims_with_exclusion(dim, exclude=None):
    dims = list(range(dim))
    if exclude is not None:
        dims.remove(exclude)

    return dims


class NormalizedFocalLossSigmoid(nn.Module):
    def __init__(
        self,
        axis=-1,
        alpha=0.25,
        gamma=2,
        max_mult=-1,
        eps=1e-12,
        from_sigmoid=False,
        detach_delimeter=True,
        batch_axis=0,
        weight=None,
        size_average=True,
        ignore_label=-1,
    ):
        super().__init__()
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

        self._from_logits = from_sigmoid
        self._eps = eps
        self._size_average = size_average
        self._detach_delimeter = detach_delimeter
        self._max_mult = max_mult
        self._k_sum = 0
        self._m_max = 0

    def forward(self, pred, label):
        pred = pred.float()
        label = label.float()

        one_hot = label > 0.5
        sample_weight = label != self._ignore_label

        if not self._from_logits:
            pred = torch.sigmoid(pred)

        alpha = torch.where(
            one_hot,
            self._alpha * sample_weight,
            (1 - self._alpha) * sample_weight,
        )
        pt = torch.where(
            sample_weight, 1.0 - torch.abs(label - pred), torch.ones_like(pred)
        )

        beta = (1 - pt) ** self._gamma

        sw_sum = torch.sum(sample_weight, dim=(-2, -1), keepdim=True)
        beta_sum = torch.sum(beta, dim=(-2, -1), keepdim=True)
        mult = sw_sum / (beta_sum + self._eps)
        if self._detach_delimeter:
            mult = mult.detach()
        beta = beta * mult
        if self._max_mult > 0:
            beta = torch.clamp_max(beta, self._max_mult)

        with torch.no_grad():
            ignore_area = (
                torch.sum(
                    label == self._ignore_label,
                    dim=tuple(range(1, label.dim())),
                )
                .cpu()
                .numpy()
            )
            sample_mult = (
                torch.mean(mult, dim=tuple(range(1, mult.dim()))).cpu().numpy()
            )
            if np.any(ignore_area == 0):
                self._k_sum = (
                    0.9 * self._k_sum
                    + 0.1 * sample_mult[ignore_area == 0].mean()
                )

                beta_pmax, _ = torch.flatten(beta, start_dim=1).max(dim=1)
                beta_pmax = beta_pmax.mean().item()
                self._m_max = 0.8 * self._m_max + 0.2 * beta_pmax

        loss = (
            -alpha
            * beta
            * torch.log(
                torch.min(
                    pt + self._eps,
                    torch.ones(1, dtype=torch.float).to(pt.device),
                )
            )
        )
        loss = self._weight * (loss * sample_weight)

        if self._size_average:
            bsum = torch.sum(
                sample_weight,
                dim=get_dims_with_exclusion(
                    sample_weight.dim(), self._batch_axis
                ),
            )
            loss = torch.sum(
                loss, dim=get_dims_with_exclusion(loss.dim(), self._batch_axis)
            ) / (bsum + self._eps)
        else:
            loss = torch.sum(
                loss, dim=get_dims_with_exclusion(loss.dim(), self._batch_axis)
            )

        return loss

    def log_states(self, sw, name, global_step):
        sw.add_scalar(
            tag=name + "_k", value=self._k_sum, global_step=global_step
        )
        sw.add_scalar(
            tag=name + "_m", value=self._m_max, global_step=global_step
        )


class FocalLoss(nn.Module):
    def __init__(
        self,
        axis=-1,
        alpha=0.25,
        gamma=2,
        from_logits=False,
        batch_axis=0,
        weight=None,
        num_class=None,
        eps=1e-9,
        size_average=True,
        scale=1.0,
        ignore_label=-1,
    ):
        super(FocalLoss, self).__init__()
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

        self._scale = scale
        self._num_class = num_class
        self._from_logits = from_logits
        self._eps = eps
        self._size_average = size_average

    def forward(self, pred, label, sample_weight=None):
        one_hot = label > 0.5
        sample_weight = label != self._ignore_label

        if not self._from_logits:
            pred = torch.sigmoid(pred)

        alpha = torch.where(
            one_hot,
            self._alpha * sample_weight,
            (1 - self._alpha) * sample_weight,
        )
        pt = torch.where(
            sample_weight, 1.0 - torch.abs(label - pred), torch.ones_like(pred)
        )

        beta = (1 - pt) ** self._gamma

        loss = (
            -alpha
            * beta
            * torch.log(
                torch.min(
                    pt + self._eps,
                    torch.ones(1, dtype=torch.float).to(pt.device),
                )
            )
        )
        loss = self._weight * (loss * sample_weight)

        if self._size_average:
            tsum = torch.sum(
                sample_weight,
                dim=get_dims_with_exclusion(label.dim(), self._batch_axis),
            )
            loss = torch.sum(
                loss,
                dim=get_dims_with_exclusion(loss.dim(), self._batch_axis),
            ) / (tsum + self._eps)
        else:
            loss = torch.sum(
                loss,
                dim=get_dims_with_exclusion(loss.dim(), self._batch_axis),
            )

        return self._scale * loss


class SoftIoU(nn.Module):
    def __init__(self, from_sigmoid=False, ignore_label=-1):
        super().__init__()
        self._from_sigmoid = from_sigmoid
        self._ignore_label = ignore_label

    def forward(self, pred, label):
        label = label.view(pred.size())
        sample_weight = label != self._ignore_label

        if not self._from_sigmoid:
            pred = torch.sigmoid(pred)

        loss = 1.0 - torch.sum(pred * label * sample_weight, dim=(1, 2, 3)) / (
            torch.sum(torch.max(pred, label) * sample_weight, dim=(1, 2, 3))
            + 1e-8
        )

        return loss


class SigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(
        self, from_sigmoid=False, weight=None, batch_axis=0, ignore_label=-1
    ):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()
        self._from_sigmoid = from_sigmoid
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

    def forward(self, pred, label):
        label = label.view(pred.size())
        sample_weight = label != self._ignore_label
        label = torch.where(sample_weight, label, torch.zeros_like(label))

        if not self._from_sigmoid:
            loss = (
                torch.relu(pred) - pred * label + F.softplus(-torch.abs(pred))
            )
        else:
            eps = 1e-12
            loss = -(
                torch.log(pred + eps) * label
                + torch.log(1.0 - pred + eps) * (1.0 - label)
            )

        loss = self._weight * (loss * sample_weight)
        return torch.mean(
            loss,
            dim=get_dims_with_exclusion(loss.dim(), self._batch_axis),
        )


def f_score(precision, recall, beta=1):
    """calculate the f-score value.

    Args:
        precision (float | torch.Tensor): The precision value.
        recall (float | torch.Tensor): The recall value.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.

    Returns:
        [torch.tensor]: The f-score value.
    """
    score = (
        (1 + beta**2)
        * (precision * recall)
        / ((beta**2 * precision) + recall)
    )
    return score


def intersect_and_union(
    pred_label,
    label,
    num_classes,
    ignore_index,
    label_map={},
    reduce_zero_label=False,
):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray | str): Prediction segmentation map
            or predict result filename.
        label (ndarray | str): Ground truth segmentation map
            or label filename.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. The parameter will
            work only when label is str. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. The parameter will
            work only when label is str. Default: False.

     Returns:
         torch.Tensor: The intersection of prediction and ground truth
            histogram on all classes.
         torch.Tensor: The union of prediction and ground truth histogram on
            all classes.
         torch.Tensor: The prediction histogram on all classes.
         torch.Tensor: The ground truth histogram on all classes.
    """

    # modified so it works with arrays
    pred_label = (
        torch.from_numpy(pred_label)
        if isinstance(pred_label, np.ndarray)
        else pred_label
    )
    label = torch.from_numpy(label) if isinstance(label, np.ndarray) else label

    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    assert {int(e) for e in set(torch.unique(label))}.issubset(
        {0, 1}
    ), "`label` should be binary"
    assert {int(e) for e in set(torch.unique(pred_label))}.issubset(
        {0, 1}
    ), "`pred_label` should be binary"

    mask = label == 1
    pred_at_mask = pred_label[mask]
    intersect = pred_at_mask[pred_at_mask == 1]

    area_intersect = intersect.sum()
    area_pred_label = pred_label.sum()
    area_label = label.sum()
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label

    # # original impl
    # mask = (label != ignore_index)
    # pred_label = pred_label[mask]
    # label = label[mask]

    # intersect = pred_label[pred_label == label]
    # area_intersect = torch.histc(
    #     intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
    # area_pred_label = torch.histc(
    #     pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    # area_label = torch.histc(
    #     label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    # area_union = area_pred_label + area_label - area_intersect


def total_intersect_and_union(
    results,
    gt_seg_maps,
    num_classes,
    ignore_index,
    label_map=None,
    reduce_zero_label=False,
):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
            truth segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """
    device = results.device
    total_area_intersect = torch.zeros(
        (num_classes,), dtype=torch.float64, device=device
    )
    total_area_union = torch.zeros(
        (num_classes,), dtype=torch.float64, device=device
    )
    total_area_pred_label = torch.zeros(
        (num_classes,), dtype=torch.float64, device=device
    )
    total_area_label = torch.zeros(
        (num_classes,), dtype=torch.float64, device=device
    )
    for result, gt_seg_map in zip(results, gt_seg_maps):
        (
            area_intersect,
            area_union,
            area_pred_label,
            area_label,
        ) = intersect_and_union(
            result,
            gt_seg_map,
            num_classes,
            ignore_index,
            label_map,
            reduce_zero_label,
        )
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return (
        total_area_intersect,
        total_area_union,
        total_area_pred_label,
        total_area_label,
    )


def mean_iou(
    results,
    gt_seg_maps,
    num_classes,
    ignore_index,
    nan_to_num=None,
    label_map={},
    reduce_zero_label=False,
):
    """Calculate Mean Intersection and Union (mIoU)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
        dict[str, float | ndarray]:
            <aAcc> float: Overall accuracy on all images.
            <Acc> ndarray: Per category accuracy, shape (num_classes, ).
            <IoU> ndarray: Per category IoU, shape (num_classes, ).
    """
    iou_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=["mIoU"],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label,
    )
    return iou_result


def mean_dice(
    results,
    gt_seg_maps,
    num_classes,
    ignore_index,
    nan_to_num=None,
    label_map=dict(),
    reduce_zero_label=False,
):
    """Calculate Mean Dice (mDice)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
        dict[str, float | ndarray]: Default metrics.
            <aAcc> float: Overall accuracy on all images.
            <Acc> ndarray: Per category accuracy, shape (num_classes, ).
            <Dice> ndarray: Per category dice, shape (num_classes, ).
    """

    dice_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=["mDice"],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label,
    )
    return dice_result


def mean_fscore(
    results,
    gt_seg_maps,
    num_classes,
    ignore_index,
    nan_to_num=None,
    label_map=dict(),
    reduce_zero_label=False,
    beta=1,
):
    """Calculate Mean Intersection and Union (mIoU)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.


     Returns:
        dict[str, float | ndarray]: Default metrics.
            <aAcc> float: Overall accuracy on all images.
            <Fscore> ndarray: Per category recall, shape (num_classes, ).
            <Precision> ndarray: Per category precision, shape (num_classes, ).
            <Recall> ndarray: Per category f-score, shape (num_classes, ).
    """
    fscore_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=["mFscore"],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label,
        beta=beta,
    )
    return fscore_result


def eval_metrics(
    results,
    gt_seg_maps,
    num_classes,
    ignore_index,
    metrics=["mIoU"],
    nan_to_num=None,
    label_map=dict(),
    reduce_zero_label=False,
    beta=1,
):
    """Calculate evaluation metrics
    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
            truth segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """

    (
        total_area_intersect,
        total_area_union,
        total_area_pred_label,
        total_area_label,
    ) = total_intersect_and_union(
        results,
        gt_seg_maps,
        num_classes,
        ignore_index,
        label_map,
        reduce_zero_label,
    )
    ret_metrics = total_area_to_metrics(
        total_area_intersect,
        total_area_union,
        total_area_pred_label,
        total_area_label,
        metrics,
        nan_to_num,
        beta,
    )

    return ret_metrics


def pre_eval_to_metrics(
    pre_eval_results, metrics=["mIoU"], nan_to_num=None, beta=1
):
    """Convert pre-eval results to metrics.

    Args:
        pre_eval_results (list[tuple[torch.Tensor]]): per image eval results
            for computing evaluation metric
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """

    # convert list of tuples to tuple of lists, e.g.
    # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
    # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
    pre_eval_results = tuple(zip(*pre_eval_results))
    assert len(pre_eval_results) == 4

    total_area_intersect = sum(pre_eval_results[0])
    total_area_union = sum(pre_eval_results[1])
    total_area_pred_label = sum(pre_eval_results[2])
    total_area_label = sum(pre_eval_results[3])

    ret_metrics = total_area_to_metrics(
        total_area_intersect,
        total_area_union,
        total_area_pred_label,
        total_area_label,
        metrics,
        nan_to_num,
        beta,
    )

    return ret_metrics


def total_area_to_metrics(
    total_area_intersect,
    total_area_union,
    total_area_pred_label,
    total_area_label,
    metrics=["mIoU"],
    nan_to_num=None,
    beta=1,
):
    """Calculate evaluation metrics
    Args:
        total_area_intersect (ndarray): The intersection of prediction and
            ground truth histogram on all classes.
        total_area_union (ndarray): The union of prediction and ground truth
            histogram on all classes.
        total_area_pred_label (ndarray): The prediction histogram on all
            classes.
        total_area_label (ndarray): The ground truth histogram on all classes.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ["mIoU", "mDice", "mFscore"]
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError("metrics {} is not supported".format(metrics))

    all_acc = total_area_intersect.sum() / total_area_label.sum()
    ret_metrics = OrderedDict({"aAcc": all_acc})
    for metric in metrics:
        if metric == "mIoU":
            iou = total_area_intersect / total_area_union
            acc = total_area_intersect / total_area_label
            ret_metrics["IoU"] = iou
            ret_metrics["Acc"] = acc
        elif metric == "mDice":
            dice = (
                2
                * total_area_intersect
                / (total_area_pred_label + total_area_label)
            )
            acc = total_area_intersect / total_area_label
            ret_metrics["Dice"] = dice
            ret_metrics["Acc"] = acc
        elif metric == "mFscore":
            precision = total_area_intersect / total_area_pred_label
            recall = total_area_intersect / total_area_label
            f_value = torch.tensor(
                [f_score(x[0], x[1], beta) for x in zip(precision, recall)]
            )
            ret_metrics["Fscore"] = f_value
            ret_metrics["Precision"] = precision
            ret_metrics["Recall"] = recall

    ret_metrics = {metric: value for metric, value in ret_metrics.items()}
    if nan_to_num is not None:
        ret_metrics = OrderedDict(
            {
                metric: np.nan_to_num(metric_value, nan=nan_to_num)
                for metric, metric_value in ret_metrics.items()
            }
        )
    return ret_metrics


# custom


def mse(output, gt_mask):
    """mean squared error"""
    return ((output - gt_mask) ** 2).sum() / output.numel()
