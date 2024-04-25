import os, sys
import torch, json
import numpy as np

from tqdm import tqdm
import cv2
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from PIL import Image
import glob
from tqdm import tqdm


def create_pred_data(preds_data_path, gt_path):
    pred_list = []
    predictions = os.listdir(preds_data_path)
    for j,pred_file in enumerate(predictions):
        item_info = {}
        preds_path = os.path.join(preds_data_path, pred_file)
        preds = torch.tensor(np.loadtxt(preds_path))
        if(len(preds.shape)==1): preds = preds.unsqueeze(0)
        if(preds.shape[1]!=0):
            output = {"boxes": preds[:,:4], 
                    "scores": preds[:,4],
                    "labels": torch.zeros(preds.shape[0])}
        else:
            output = {"boxes": torch.tensor([[0,0,0,0]]), 
                    "scores": torch.tensor([0]),
                    "labels": torch.zeros(0)}
            # import pdb; pdb.set_trace()
        target_path = os.path.join(gt_path, pred_file.replace("_preds", ""))
        if(os.path.isfile(target_path)):
            targets = torch.tensor(np.loadtxt(target_path))
            if(targets.shape[0]!=0):
                if (len(targets.shape)==1): targets = targets.unsqueeze(0)
                targets=targets[:,1:] 
            else:
                print(folder)
                targets=torch.tensor([])
        else:
            targets = torch.tensor([])
        item_info['pred'] = output
        item_info['target'] = {"boxes":targets}
        pred_list.append(item_info)
    return pred_list


def get_confmat(pred_list, threshold = 0.3):
    def true_positive(gt, pred):
        # If center of pred is inside the gt, it is a true positive
        box_pascal_gt = ( gt[0]-(gt[2]/2.) , gt[1]-(gt[3]/2.), gt[0]+(gt[2]/2.), gt[1]+(gt[3]/2.) )
        if (pred[0] >= box_pascal_gt[0] and pred[0] <= box_pascal_gt[2] and
                pred[1] >= box_pascal_gt[1] and pred[1] <= box_pascal_gt[3]):
            return True
        return False

    #tp, tn, fp, fn
    conf_mat = np.zeros((4))
    error_image = np.zeros((len(pred_list)))
    conf_mat_idx = []
    for i, data_item in enumerate(pred_list):
        gt_data = data_item['target']
        pred = data_item['pred']
        scores = pred['scores']
        select_mask = scores > threshold
        pred_boxes = pred['boxes'][select_mask]
        out_array = np.zeros((4))
        for j, gt_box in enumerate(gt_data['boxes']):
            add_tp = False
            new_preds = []
            for pred in pred_boxes:
                if true_positive(gt_box, pred):
                    add_tp = True
                else:
                    new_preds.append(pred)
            pred_boxes = new_preds
            if add_tp:
                out_array[0] += 1
            else:
                out_array[3] += 1
        out_array[2] = len(pred_boxes)
        conf_mat+=out_array
        conf_mat_idx.append(out_array)
        if(out_array[2]!=0 or out_array[3]!=0):
            error_image[i] = 1
    return conf_mat, error_image, conf_mat_idx



def calc_froc(pred_data, fps_req = [0.025,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1,1.1,1.2,1.5,1.8,1.9,2,2.4,2.7,3,4.4,5.4], num_thresh = 1000):
    num_images = len(pred_data)
    # fps_req = np.linspace(0,2,num_thresh)
    thresholds = np.linspace(0,1,num_thresh)
    conf_mat_thresh = np.zeros((num_thresh, 4))
    for i, thresh_val in enumerate( tqdm(thresholds) ):
        conf_mat,_,_ = get_confmat(pred_data, thresh_val)
        conf_mat_thresh[i] = conf_mat
    
    sensitivity = np.zeros((num_thresh)) #recall
    for i in range(num_thresh):
        conf_mat = conf_mat_thresh[i]
        if((conf_mat[0]+conf_mat[3])==0):
            sensitivity[i] = 0
        else:
            sensitivity[i] = conf_mat[0]/(conf_mat[0]+conf_mat[3])
    
    froc_fpis =[]
    froc_sens =[]    
    for fp_req in fps_req:
        for i in range(num_thresh):
            f = conf_mat_thresh[i][2]
            if f/num_images < fp_req:                
                froc_sens.append(sensitivity[i-1])
                froc_fpis.append(conf_mat_thresh[i-1][2]/num_images)
                fp_req = f"{fp_req:.{1}f}"
                sensitivity[i-1] = f"{sensitivity[i-1]:.{5}f}"
                thresholds[i] = f"{thresholds[i]:.{5}f}"
                print(f"FPI: {fp_req}, Sensitivity: {sensitivity[i-1]}, Threshold: {thresholds[i]}")
                break
    return froc_fpis, froc_sens


if __name__ == '__main__':
    PREDS_PATH = "test/predictions"
    GT_PATH = "test/labels"
    prediction_dict = create_pred_data(PREDS_PATH, GT_PATH)
    calc_froc(prediction_dict)
    
