"""
author: Timothy C. Arlen
date: 28 Feb 2018

Calculate Mean Average Precision (mAP) for a set of bounding boxes corresponding to specific
image Ids. Usage:

> python calculate_mean_ap.py

Will display a plot of precision vs recall curves at 10 distinct IoU thresholds as well as output
summary information regarding the average precision and mAP scores.

NOTE: Requires the files `ground_truth_boxes.json` and `predicted_boxes.json` which can be
downloaded fromt this gist.
"""

from __future__ import absolute_import, division, print_function

from copy import deepcopy
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import argparse
import pickle
import sys
from nms import nms


parser = argparse.ArgumentParser()
parser.add_argument('--show_images', action='store_true')
parser.add_argument('--reset', action='store_true')
parser.add_argument('--cvwait', type=int, default=0)
parser.add_argument('--nms', type=float, default=0)
args = parser.parse_args()


# import seaborn as sns
# sns.set_style('white')
# sns.set_context('poster')

COLORS = [
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
    '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
    '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
    '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']


def do_nms(boxes, scores):
    bboxes = []
    for b in boxes:
        x = (b[0]) * 1024
        y = (b[1]) * 768
        w = (b[2] - b[0]) * 1024
        h = (b[3] - b[1]) * 768
        bboxes.append([x, y, w, h])

    l = nms.boxes(bboxes, scores, score_threshold=1, nms_threshold=args.nms)
    boxes[:] = [boxes[int(i)] for i in l]
    scores[:] = [scores[int(i)] for i in l]


def calc_iou_individual(pred_box, gt_box):
    """Calculate IoU of single predicted and ground truth box

    Args:
        pred_box (list of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (list of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

    Returns:
        float: value of the IoU for the two boxes.

    Raises:
        AssertionError: if the box is obviously malformed
    """
    x1_t, y1_t, x2_t, y2_t = gt_box
    x1_p, y1_p, x2_p, y2_p = pred_box

    if (x1_p > x2_p) or (y1_p > y2_p):
        raise AssertionError(
            "Prediction box is malformed? pred box: {}".format(pred_box))
    if (x1_t > x2_t) or (y1_t > y2_t):
        raise AssertionError(
            "Ground Truth box is malformed? true box: {}".format(gt_box))

    if (x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t):
        return 0.0

    far_x = np.min([x2_t, x2_p])
    near_x = np.max([x1_t, x1_p])
    far_y = np.min([y2_t, y2_p])
    near_y = np.max([y1_t, y1_p])

    inter_area = (far_x - near_x) * (far_y - near_y)
    true_box_area = (x2_t - x1_t) * (y2_t - y1_t)
    pred_box_area = (x2_p - x1_p) * (y2_p - y1_p)
    iou = inter_area / (true_box_area + pred_box_area - inter_area)
    return iou


def show_images(gt, predictions, pred_match_idx, ious_pbox, img_id, tp, fp, fn):
    scores = predictions['scores']
    pred_boxes = predictions['boxes']
    gt_boxes = gt['boxes']
    difficult = gt['difficult']

    impath = img_id.split("_SrcImages_")
    impath = os.path.join('/home/honza/dev/_videa/', impath[0], "SrcImages", impath[1])
    # impath = os.path.join('/home/honza/dev/_train_data/openimages/', impath[0], "SrcImages", impath[1])
    # img = np.ones((480, 640, 3), np.uint8) * 255
    if not os.path.exists(impath):
        return
    img = cv2.imread(impath)
    h, w, c = img.shape
    cv2.putText(img, "{} tp: {} fp: {} fn: {}".format(img_id, tp, fp, fn), (5, 5 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), lineType=cv2.LINE_AA)
    for i, b in enumerate(gt_boxes):
        x1 = int(round(b[0] * w))
        y1 = int(round(b[1] * h))
        x2 = int(round(b[2] * w))
        y2 = int(round(b[3] * h))
        color = (0, 255, 0)
        if difficult[i]:
            color = (0, 255, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    for i, b in enumerate(pred_boxes):
        x1 = int(round(b[0] * w))
        y1 = int(round(b[1] * h))
        x2 = int(round(b[2] * w))
        y2 = int(round(b[3] * h))
        color = (128, 128, 0)
        if i in pred_match_idx:
            color = (0, 0, 255)
            cv2.putText(img, "{}".format(int(round(ious_pbox[i] * 100))), (x1, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, lineType=cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        cv2.putText(img, "{}".format(((scores[i]))), (max(0, x1), y2 + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, lineType=cv2.LINE_AA)

    cv2.imshow('test', img)
    k = cv2.waitKey(args.cvwait)
    if k == ord('q') or k == 27:
        sys.exit(0)


def get_single_image_results(gt, predictions, iou_thr, img_id=None):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.

    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`)
            and 'scores'
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.

    Returns:
        dict: true positives (int), false positives (int), false negatives (int)
    """

    scores = predictions['scores']
    pred_boxes = predictions['boxes']
    gt_boxes = gt['boxes']
    difficult = gt['difficult']

    all_pred_indices = range(len(pred_boxes))
    all_gt_indices = range(len(gt_boxes))
    if len(all_pred_indices) == 0:
        tp = 0
        fp = 0
        fn = len(gt_boxes)
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn, 'avg_iou': 0.0}
    if len(all_gt_indices) == 0:
        tp = 0
        fp = len(pred_boxes)
        fn = 0
        if args.show_images:
            show_images(gt, predictions, [], [], img_id, tp, fp, fn)
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn, 'avg_iou': 0.0}

    gt_idx_thr = []
    pred_idx_thr = []
    ious = []
    ious_pbox = {}
    for ipb, pred_box in enumerate(pred_boxes):
        ious_pbox[ipb] = 0.0
        for igb, gt_box in enumerate(gt_boxes):
            iou = calc_iou_individual(pred_box, gt_box)
            if iou > iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)
                if ipb not in ious_pbox or ious_pbox[ipb] < iou:
                    ious_pbox[ipb] = iou

    args_desc = np.argsort(ious)[::-1]
    if len(args_desc) == 0:
        # No matches
        tp = 0
        fp = len(pred_boxes)
        fn = len(gt_boxes) - sum(difficult)
        avg_iou = 0.0
    else:
        gt_match_idx = []
        pred_match_idx = []
        difficult_matched = 0
        for idx in args_desc:
            gt_idx = gt_idx_thr[idx]
            pr_idx = pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
                if difficult[gt_idx]:
                    difficult_matched = difficult_matched + 1

        tp = len(gt_match_idx)
        fp = len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - sum(difficult) + difficult_matched - len(gt_match_idx)

        avg_iou = sum(ious) / len(ious)

        if args.show_images:
            show_images(gt, predictions, pred_match_idx, ious_pbox, img_id, tp, fp, fn)

    return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn, 'avg_iou': avg_iou}


def calc_precision_recall(img_results):
    """Calculates precision and recall from the set of images

    Args:
        img_results (dict): dictionary formatted like:
            {
                'img_id1': {'true_pos': int, 'false_pos': int, 'false_neg': int},
                'img_id2': ...
                ...
            }

    Returns:
        tuple: of floats of (precision, recall)
    """
    true_pos = 0
    false_pos = 0
    false_neg = 0
    for _, res in img_results.items():
        true_pos += res['true_pos']
        false_pos += res['false_pos']
        false_neg += res['false_neg']

    try:
        precision = true_pos / (true_pos + false_pos)
    except ZeroDivisionError:
        precision = 0.0
    try:
        recall = true_pos / (true_pos + false_neg)
    except ZeroDivisionError:
        recall = 0.0

    return (precision, recall)


def get_model_scores_map(pred_boxes):
    """Creates a dictionary of from model_scores to image ids.

    Args:
        pred_boxes (dict): dict of dicts of 'boxes' and 'scores'

    Returns:
        dict: keys are model_scores and values are image ids (usually filenames)

    """

    minsc = 10000
    maxsc = 0
    for img_id, val in pred_boxes.items():
        for score in val['scores']:
            if score > maxsc:
                maxsc = score
            if score < minsc:
                minsc = score

    pts = np.linspace(minsc, maxsc, 100)

    model_scores_map = {}
    for img_id, val in pred_boxes.items():
        for score in val['scores']:
            # ind = np.searchsorted(pts, score, side='left')
            # key = pts[ind]
            key = score
            if key not in model_scores_map.keys():
                model_scores_map[key] = [img_id]
            else:
                model_scores_map[key].append(img_id)
    return model_scores_map


def get_avg_precision_at_iou(gt_boxes, pred_boxes, iou_thr=0.5):
    """Calculates average precision at given IoU threshold.

    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (list of list of floats): list of locations of predicted
            objects as [xmin, ymin, xmax, ymax]
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.

    Returns:
        dict: avg precision as well as summary info about the PR curve

        Keys:
            'avg_prec' (float): average precision for this IoU threshold
            'precisions' (list of floats): precision value for the given
                model_threshold
            'recall' (list of floats): recall value for given
                model_threshold
            'models_thrs' (list of floats): model threshold value that
                precision and recall were computed for.
    """

    if args.nms != 0:
        for key in pred_boxes:
            do_nms(pred_boxes[key]['boxes'], pred_boxes[key]['scores'])

    model_scores_map = get_model_scores_map(pred_boxes)
    sorted_model_scores = sorted(model_scores_map.keys())

    # Sort the predicted boxes in descending order (lowest scoring boxes first):
    for img_id in pred_boxes.keys():
        arg_sort = np.argsort(pred_boxes[img_id]['scores'])
        pred_boxes[img_id]['scores'] = np.array(pred_boxes[img_id]['scores'])[arg_sort].tolist()
        pred_boxes[img_id]['boxes'] = np.array(pred_boxes[img_id]['boxes'])[arg_sort].tolist()

    pred_boxes_pruned = deepcopy(pred_boxes)

    all_img_ids = sorted(set().union(pred_boxes.keys(), gt_boxes.keys()))
    # all_img_ids = sorted(set().union(pred_boxes.keys()))

    avg_ious = []
    precisions = []
    recalls = []
    model_thrs = []
    img_results = {}
    # Loop over model score thresholds and calculate precision, recall
    for ithr, model_score_thr in enumerate(sorted_model_scores[:-1]):
        # On first iteration, define img_results for the first time:
        img_ids = all_img_ids if ithr == 0 else model_scores_map[model_score_thr]
        for i, img_id in enumerate(img_ids):
            if img_id not in gt_boxes:
                gt = {'boxes': [], 'difficult': [], 'labels': []}
                print('no gt for ' + img_id)
            else:
                gt = gt_boxes[img_id]

            if img_id not in pred_boxes_pruned:
                img_results[img_id] = get_single_image_results(gt, {'boxes': [], 'scores': []}, iou_thr, img_id)
                # print('no detections in ' + img_id)
                continue
            else:
                box_scores = pred_boxes_pruned[img_id]['scores']
            start_idx = 0
            for score in box_scores:
                if score <= model_score_thr:
                    pred_boxes_pruned[img_id]
                    start_idx += 1
                else:
                    break
            # print("{} {}".format(i, img_id))

            # Remove boxes, scores of lower than threshold scores:
            pred_boxes_pruned[img_id]['scores'] = pred_boxes_pruned[img_id]['scores'][start_idx:]
            pred_boxes_pruned[img_id]['boxes'] = pred_boxes_pruned[img_id]['boxes'][start_idx:]

            # Recalculate image results for this image
            img_results[img_id] = get_single_image_results(gt, pred_boxes_pruned[img_id], iou_thr, img_id)

        # avg_iou = sum([x['avg_iou'] for x in img_results]) / len(img_results)
        avg_iou = 0.0

        prec, rec = calc_precision_recall(img_results)
        precisions.append(prec)
        recalls.append(rec)
        model_thrs.append(model_score_thr)
        avg_ious.append(img_results[img_id]['avg_iou'])

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    prec_at_rec = []
    for recall_level in np.linspace(0.0, 1.0, 11):
        try:
            rargs = np.argwhere(recalls >= recall_level).flatten()
            prec = max(precisions[rargs])
        except ValueError:
            prec = 0.0
        prec_at_rec.append(prec)
    avg_prec = np.mean(prec_at_rec)

    avg_iou = sum(avg_ious) / len(avg_ious)

    return {
        'avg_prec': avg_prec,
        'precisions': precisions,
        'recalls': recalls,
        'model_thrs': model_thrs,
        'avg_iou': avg_iou}


def plot_pr_curve(precisions, recalls, category='Person', label=None, color=None, ax=None, title=''):
    """Simple plotting helper function"""

    if ax is None:
        plt.figure(figsize=(10, 8))
        ax = plt.gca()

    if color is None:
        color = COLORS[0]
    # ax.scatter(recalls, precisions, label=label, s=20, color=color)
    ax.plot(recalls, precisions, label=label, color=color, marker='')
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.set_title('Precision-Recall curve for {} {}'.format(category, title))
    return ax


def calc_results(gt_file, pred_files, thresholds=[0.2], title=''):
    with open(gt_file) as infile:
        gt_boxes = json.load(infile)

    idx = 0
    ax = [None for x in range(len(thresholds))]
    for rf in pred_files:
        with open(rf) as infile:
            pred_boxes = json.load(infile)

        avg_precs = []
        iou_thrs = []
        avg_iou = []
        for i, iou_thr in enumerate(thresholds):
            time1 = time.time()

            ofile = os.path.basename(rf)
            ofile = rf.rsplit('.', 1)[0]
            ofile = '{}_{}'.format(ofile, iou_thr)
            ofile = ofile + '.pickle'
            loaded = False
            if os.path.exists(ofile) and not args.show_images and not args.reset:
                with open(ofile, 'rb') as f:
                    data = pickle.load(f)
                loaded = True
            else:
                data = get_avg_precision_at_iou(gt_boxes, pred_boxes, iou_thr=iou_thr)
            dur = time.time() - time1
            print("execution time = {}ms".format(dur * 1000))
            avg_precs.append(data['avg_prec'])
            iou_thrs.append(iou_thr)
            avg_iou.append(data['avg_iou'])

            precisions = data['precisions']
            recalls = data['recalls']
            ax[i] = plot_pr_curve(precisions, recalls, label='{}: {:.2f}'.format(os.path.basename(rf), iou_thr),
                                  color=COLORS[idx % len(COLORS)], ax=ax[i], title=title)
            idx += 1

            if not loaded:
                with open(ofile, 'wb') as f:
                    pickle.dump(data, f)

        # prettify for printing:
        avg_precs = [float('{:.4f}'.format(ap)) for ap in avg_precs]
        iou_thrs = [float('{:.4f}'.format(thr)) for thr in iou_thrs]
        avg_iou = [float('{:.4f}'.format(iou)) for iou in avg_iou]
        print('map: {:.2f}'.format(100 * np.mean(avg_precs)))
        print('avg precs: ', avg_precs)
        print('iou_thrs:  ', iou_thrs)
        print('avg_ious:  ', avg_iou)

        # break

    for x in ax:
        x.legend(loc='lower left', title='IOU Thr', frameon=True)
        x.grid()
        x.set_xlim([0.5, 1.1])
        x.set_ylim([0.5, 1.1])

    # for xval in np.linspace(0.0, 1.0, 11):
    #     plt.vlines(xval, 0.0, 1.1, color='gray', alpha=0.3, linestyles='dashed')


if __name__ == "__main__":
    gt_file = 'gt/all_annots.json'

    results = [#'r22_person',
               # 'y',
               'r22cv1',
               'r22__person',
               # 'r9_person',
               # 'lrm7_head',
               # 'lrm8_head',
               # 'lrm9_head',
               # 'lrm9_10000head',
               # 'lrm9_20000head',
               # 'lrm9_30000head',
               # 'lrm9_40000head',
               # 'lrm9_50000head',
               # 'lrm9_50000__head',
               # 'lrm9_50000_obj_head',
               # 'lrm9_55000head',
               # 'lrm9_56000head',
               # 'lrm9_57000head',
               # 'lrm9_58000head',
               # 'lrm9_59000head',
               # 'lrm9_60000head',
               # 'lrm10_head',
               # 'lrm10_10000head',
               # 'lrm10_20000head',
               # 'lrm10_30000head',
               # 'lrm10_40000head',
               # 'lrm10_50000head',
               # 'lrm10_55000head',
               # 'lrm10_56000head',
               # 'lrm10_57000head',
               # 'lrm10_58000head',
               # 'lrm10_59000head',
               # 'lrm10_60000head',
               ]

    # calc_results(gt_file, [os.path.join('results', r + '.json') for r in results], [0.2])

    # plt.show()
    # sys.exit(0)

    videa = ['video0001',
             # 'video0002',
             # 'v8ss',
             # 'zahrada4',
             # '2017091908_tul',
             # '2017120812_2.2m_radial',
             # '2017120812_2.2m_tangens',
             # 'chairs'
             ]

    for v in videa:
        gt_file = os.path.join('gt', v + '.json')

        calc_results(gt_file, [os.path.join('results', r + '_' + v + '.json') for r in results], [0.2], title=v)

    plt.show()
