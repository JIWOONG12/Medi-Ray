from config import parse_arguments # config.py 의 parse_arguments
from datasets import DiseaseDataset # datasets.py의 DiseaseDataset
from models import vgg16
from models.inception_v3 import Inception3

import pandas as pd
from utils_folder.eval_metric import *

import pathlib

import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader
from models.resnet import resnet50

import torchvision; torchvision.version
import torchvision.models as models

def plot_roc_curves(gts, preds, num_classes, class_list, save_dir):#gts : 실제 정답

    plt.figure(figsize=(8, 5))  # 크기를 5x3 인치로 축소
    auroc_scores = []
    
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(np.array(gts)[:, i], np.array(preds)[:, i])
        roc_auc = auc(fpr, tpr)
        auroc_scores.append(roc_auc)
        plt.plot(
            fpr, 
            tpr, 
            label=f'{class_list[i]} (AUC = {roc_auc:.3f})'
        )
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=0.5)  # 대각선 선 두께 감소
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    # 폰트 크기 
    plt.xlabel('False Positive Rate', fontsize=8)
    plt.ylabel('True Positive Rate', fontsize=8)
    plt.title(os.path.basename(save_dir), fontsize=10)
    
    # 축 레이블 크기 
    plt.tick_params(axis='both', which='major', labelsize=6)
    
    # 범례 크기와 위치 조정
    plt.legend(loc="center left", bbox_to_anchor=(1,0.5),
              fontsize=8, handlelength=0.5)
    
    plt.grid(True, linewidth=0.2)  # 그리드 선 두께 감소
    plt.tight_layout()
    
    # 저장 옵션 수정
    plt.savefig(os.path.join(save_dir, 'roc_curves.png'), 
                bbox_inches='tight',
                format='png')  # 파일 크기 최적화
    plt.close()

def evaluate(args, loader, model, device, num_classes , class_list):

    model.eval() # 평가
    correct = 0
    total = 0
    overall_logits = []
    overall_preds = []
    overall_gts = []

    for iter_, (imgs, labels) in enumerate(iter(loader)): # loader에서 iter_, imags, labels 가져옴
        print(f"Initial labels: {labels}")
        imgs = imgs.to(device) # device로 imgs 전송
        labels = labels.to(device, dtype=torch.long) # device로 labels 전송, 데이터 타입은 64bit 정수

        outputs = model(imgs) # model의 결과를 outputs에 저장
        outputs = torch.sigmoid(outputs) # sigmoid의 결과를 업데이트
        outputs_preds = outputs.clone() # 텐서 복제
        overall_logits += outputs.cpu().detach().numpy().tolist() # outputs을 list 형태로 반     
        outputs_preds[outputs_preds >= 0.5] = 1 # 예측값이 0.5 이상이면 1
        outputs_preds[outputs_preds < 0.5] = 0 # 미만이면 
        print(f"outputs_preds: {outputs_preds}")
        total += labels.size(0) * labels.size(1) # 배치사이즈 * 클래스 수 저장
        correct += torch.sum(outputs_preds == labels.data).item() 

        ## For evaluation
        overall_preds += outputs_preds.cpu().detach().numpy().tolist() # outputs를 list로 반환
        overall_gts += labels.cpu().detach().numpy().tolist() # labels를 list로 반환
        
    print('[*] Test Acc: {:5f}'.format(100.*correct/total)) # 테스트 정확도 백분율로 반환

    # eval_metric의 compute_AUCs
    AUROCs = compute_AUCs(overall_gts, overall_logits, num_classes , class_list) # return 값 > 딕셔너리
    AUROC_avg = np.array(AUROCs).mean() # 배열로 변환후 평균 구함
    
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg)) # The average AUROCis {AUROC_avg}
    for i in range(args.num_class): # num_classes 만큼 반복
        # if class_list[i] == 'Fracture': # class_list[i]가 Facture이라면
        # 아니라면
        print('The AUROC of {} is {}'.format(class_list[i], AUROCs[i]))  # The AUROC of {class_list[i]} is {AUROCs[i]}
    
    print("\nPlotting ROC curves...")
    plot_roc_curves(overall_gts, overall_logits, num_classes, class_list, args.log_dir)
    print(f"ROC curves saved to {args.log_dir}/roc_curves.png")

    print("\nPlotting average AUROC...")
    plot_average_auroc(AUROCs, class_list, args.log_dir)
    print(f"Average AUROC plot saved to {args.log_dir}/average_auroc.png")
    

def main(args):
    ##### Initial Settings
    json_data = pd.read_json(args.test_path) #json
    class_list = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
                   'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                   'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
                   'Pneumonia', 'Pneumothorax', 'Support Devices'] # warning #json컬럼명 불러오기 [5:] 만큼
    print("[*] class list : " , class_list) # class_list 출력
    num_classes = args.num_class # num_classes 선언
    # downstream = '{}_{}_class'.format(args.downstream_name, num_classes) # {args.downstream_name}_{num_classes}_class

    # print('\n[*****] ', downstream)
    print('[*] using {} bit images'.format(args.bit))

    # device check & pararrel
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # cuda가 사용가능하다면 cuda, 아니라면 cpu
    print('[*] device: ', device) # device 출력

    # path setting
    pathlib.Path(args.log_dir).mkdir(parents=True, exist_ok=True) # 폴더 생성, 상위 폴더 없으면 생
    
    # for log
    f = open(os.path.join(args.log_dir,'arguments.txt'), 'w') # log_dir의 arguments.txt 파일 쓰기 권한으로 불러옴
    f.write(str(args)) # args 를 문자열로 변환해서 입력
    f.close() # 폴더 닫음
    print('[*] log directory: {} '.format(args.log_dir)) # log directory : {log_dir}
    
    if args.seed is not None: # args.seed가 None이 아니라면
        random.seed(args.seed) # randcom.seed(args.seed) : 시드값 지정
        torch.manual_seed(args.seed) # torch.manual_seed를 args.seed로 고정
        os.environ['PYTHONHASHSEED'] = str(args.seed) # os 자체의 seed 고정
        np.random.seed(args.seed) # numpy seed 고정 
        torch.cuda.manual_seed(args.seed) # cudnn seed 고정
        torch.backends.cudnn.deterministic = True # cudnn seed 고정(nn.Conv2d)
        torch.backends.cudnn.benchmark = False # CUDA 내부 연산에서 가장 빠른 알고리즘을 찾아 수행

    # select network
    print('[*] build network... backbone: {}'.format(args.backbone))
    if args.backbone == 'resnet50': # backbone 설정
        model = resnet50(num_classes=args.num_class)
    elif args.backbone == 'vgg':
        model = vgg16(num_classes=args.num_class)
    elif args.backbone == 'densenet':
        model = densenet169(num_classes=args.num_class)
    elif args.backbone == 'inception':
        model = Inception3(num_classes=args.num_class)
    elif args.backbone == 'vit':
        vit = models.vit_b_16(pretrained = True)
        head_in_features = vit.heads.head.in_features
        vit.heads.head = nn.Linear(head_in_features, 14)
        model = vit
    else:
        ValueError('Have to set the backbone network in [resnet, vgg, densenet]')

    model = model.to(device) # model을 device로 전송

    if args.resume is True: # resum이 True라면
        checkpoint = torch.load(args.pretrained) # checkpoint 를 pretraied에서 불러옴
        pretrained_dict = checkpoint['state_dict'] # checkpoint의 state_dict를 pretrained_dict에 저장
        pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()} 
        model.load_state_dict(pretrained_dict) # model에 사전학습 모듈의 state 로드
        print("load model completed")
        print("Model loaded:", args.pretrained)
    ##### Dataset & Dataloader
    print('[*] prepare datasets & dataloader...')
    test_datasets = DiseaseDataset(args.test_path, 'test', args.img_size, args.bits, args) # DiseaseDatatset함수 사용해서 데이터 불러옴
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=1,  # test_datasets의 데이터 로드
    num_workers=args.w, pin_memory=True, drop_last=True)
    
    ##### Train & Test
    print('[*] start a test')
    evaluate(args, test_loader, model, device, num_classes , class_list) # evaluate 함수 사용하여 평가
        
if __name__ == '__main__':
    argv = parse_arguments(sys.argv[1:])
    main(argv)
