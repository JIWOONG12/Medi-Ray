import sys
import os

import numpy as np
import pathlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, auc, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import itertools

import json

def compute_AUCs(gt, pred , num_classes , class_list):
    
    """
    https://github.com/arnoweng/CheXNet/blob/master/model.py
    Computes Area Under the Curve (AUC) from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = [] # 배열 AUROCs 선언
    gt_np = np.array(gt) # gt를 배열로 변환하여 gt_np에 저장
    pred_np = np.array(pred) # pred를 배열로 변환하여 pred_np에 저장
    for i in range(num_classes): # num_classes 만큼 반복
        if class_list[i] == 'Fracture': # claass_[i] 가 Fracture이라면
            continue # if 계속 진행
        # Fracutre이 아니라면
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i])) # AUROCs에 gt, pred의 AUC 점수를 추가
    return AUROCs # AUROCs 리스트가 반환


def compute_confusion_matrix(gt, preds , num_classes , class_list):
    
    conf_mat_dict={} # 딕셔너리 conf_mat_dict 선언
    gt_np = np.array(gt) # gt를 배열로 변환하여 저장
    pred_np = np.array(preds) # pred를 배열로 변환하여 저장
    for i in range(num_classes): # num_classes 만큼 반복
        y_true_label = gt_np[:, i] # y_true_label 저장
        y_pred_label = pred_np[:, i] # y_pred_label 저장
        # confusion_matrix : 혼동행렬
        # 예측한 클래스와 실제 클래스 간의 관계를 보여주는 행렬
        conf_mat_dict[class_list[i]] = confusion_matrix(y_pred=y_pred_label, y_true=y_true_label) # 예측값과 실제 레이블 추출하고 혼동행렬 계산한 후 딕셔너리에 저장
        # key = class_list[i] : 클래스명

    return conf_mat_dict

def save_confusion_matrix(cm, target_names, log_dir, title='CFMatrix', cmap=None, normalize=False):
    # cm : 혼동행렬
    # np.trace() : 혼동행렬의 대각선 요소들의 합, TP+TN : 모델이 올바르게 분류한 경우
    # np.sum(cm) : 모든 요소의 합 > 전체 데이터 샘플 수
    acc = np.trace(cm) / float(np.sum(cm)) # 올바르게 예측한 수 / 전체 샘플수
    misclass = 1 - acc # 오분류율(잘못 분류한 비율)

    # cmap : color map
    if cmap is None: # 만약 cmap이 None이라면
        cmap = plt.get_cmap('Blues') # Blues라는 cmap 가져온다

    plt.figure(figsize=(12,10)) # 그래프 사이즈 지정
    # imterpolation : 픽셀 사이의 보간 방법 > 이미지를 확대하거나 축소할때 픽셀값 사이의 색을 어떻게 설정할지..
    # nearest : 가장 가까운 픽셀 값을 가져옴
    plt.imshow(cm, interpolation='nearest', cmap=cmap) # 그래프 그리기
    plt.title(title) # 타이틀 지정
    plt.colorbar() # 컬러바 표현

    if target_names is not None: # 만약 target_names가 None이라면
        tick_marks = np.arange(len(target_names)) # tick_marks는 target_names의 길이 배열
        plt.xticks(tick_marks, target_names, rotation=45) # x축 설정, rotation=45 : 레이블 회전시켜 가독석을 높임
        plt.yticks(tick_marks, target_names) # y축 설정

    if normalize: # Normalize 가 True인 경우 진행
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # 혼동행렬
        # cm을 float 형으로 변환
        # cm.sum(axis=1) : 혼동행렬에 대한 합계 계산
        # [: np.newaxis] : 결과를 2D 배열로 변환..?
        # 혼동행렬 정규화

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2 #
    # cm.max() : 혼동행렬 최대값
    # 정규화 True인 경우 최대값을 1.5로 나누고 아닌 경우 2로 나눠서 thresh 설정
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])): # i : shape[0], j : shape[1]
        if normalize: # 정규화 = True
            plt.text(j, i, "{:0.4f}".format(cm[i, j]), # 텍스트 표시, 소숫점 4자리까지
                    horizontalalignment="center", # 중앙에
                    color="white" if cm[i,j] > thresh else "black") # thresh보다 크면 흰색, 아니면 검은색
        else: # 정규화 = False
            plt.text(j, i, "{:,}".format(cm[i, j]), # 텍스트 표시, 정수로
                    horizontalalignment="center", # 중앙에
                    color="white" if cm[i,j] > thresh else "black") # thresh 보다 크면 흰색, 아니면 검은색

    plt.tight_layout() # 레이아웃 자동 조정
    plt.ylabel('True label') # y축 라벨
    plt.xlabel('Predicted label\n accuracy={:0.4f}'.format(acc)) # x축 라벨
    plt.savefig(os.path.join(log_dir, 'confusion_matrix.png')) # png로 저장
    
def get_mertrix(confusion_matrix, log_dir, class_list=['Normal', 'Abnormal']):
    
    cnf_matrix = confusion_matrix # 혼동행렬
    save_confusion_matrix(cnf_matrix, class_list, log_dir) # 혼동행렬 저장?
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  # FP : False Positive
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix) # FN : False Nagative
    TP = np.diag(cnf_matrix) # TP : True Positive
    TN = cnf_matrix.sum() - (FP + FN + TP) # TN : True Nagative
    FP = FP.astype(float) # FP 타입 float로
    FN = FN.astype(float) # FN 타입 float로
    TP = TP.astype(float) # TP 타입 float로
    TN = TN.astype(float) # TN 타입 float로
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN) # TPR : True 중 실제로 True로 예측된 비율
    # Specificity or true negative rate
    TNR = TN/(TN+FP) # TNR : False 중 False로 예측된 비율
    # Precision or positive predictive value
    PPV = TP/(TP+FP) # True로 예측한 것 중 실제로 True인 비율
    # Negative predictive value
    NPV = TN/(TN+FN) # False로 예측한 것중 실제로 False인 비율
    # Fall out or false positive rate
    FPR = FP/(FP+TN) # 실제는 False인데 True로 예측한 비율
    # False negative rate
    FNR = FN/(TP+FN) # 실제는 Ture인데 False로 예측한 비율
    # False discovery rate
    FDR = FP/(TP+FP) # True로 예측한 것중 False인 비율
    F1_Score = 2*(PPV*TPR) / (PPV+TPR) # F1 Score : 정밀도와 재현률의 조화평균
    # Overall accuracy for each class
    ACC = (TP+TN)/(TP+FP+FN+TN) # 정확도 > 올바르게 예측한 비율


    print('specificity: ', TNR) # 특이도 : False를 정확히 예측한 비율
    print('sensitivity (recall): ', TPR) # true positive rate
    print('positive predictive value (precision): ', PPV) # True로 예측한 것중 True인 비율
    print('negative predictive value: ', NPV) # False로 예측한 것중 False인 비율
    print('ACC: ', ACC) # 정확도
    print('F1_score: ', F1_Score) # F1 Score

