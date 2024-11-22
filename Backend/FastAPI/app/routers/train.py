import os
import sys
import random
import time
import pathlib
import numpy as np
import pandas as pd
from config import parse_arguments # config.py의 parse_arguments
from datasets import DiseaseDataset # datasets.py의 DiseaseDataset

from models.resnet import resnet50
from models import vgg16
from models.inception_v3 import Inception3

from utils_folder.utils import AverageMeter, ProgressMeter, save_model # utils_folder/utils.py의 함수들

import torch
import torch.nn as nn

from tensorboardX import SummaryWriter

def calculate_parameters(model):
    # model의 파라미터 반환
    # numel()은 파라미터 텐서의 총 요소 개수 반환
    # sum() 파라미터 값의 합을 반환
    return sum(param.numel() for param in model.parameters())/1000000.0

# def get_class_weights(labels_dict,device):
#    total_samples = sum(len(v[0]) + len(v[1]) for v in labels_dict.items())
#    weights = []
#    for class_counts in labels_dict.values():
#        pos_samples = class_counts[1]
#        neg_samples = class_counts[0]
#        weight = (total_samples / 2.0) / pos_samples  # 양성 클래스에 대한 가중치
#        weights.append(weight)
#    return torch.FloatTensor(weights).to(device)
    
def train(args, epoch, loader, val_loader, model, device, optimizer, writer ,scheduler):

    model.train() # 모델 학습
    batch_time = AverageMeter('Time', ':6.3f') # name은 Time, format 은 6.3f > 전체 6자리, 소수점 3자리까지 출력
    losses = AverageMeter('Loss', ':.4f') # 이름 Loss, format은 .4f > 소수점 4자리까지 출력
    # Utils의 ProgressMeter
    progress = ProgressMeter(
        len(loader), # loader의 길이 > 배치수 # loader은 train_loader
        [batch_time, losses], # 모니터링할 지표
        prefix='Epoch: [{}]'.format(epoch)) # Epoch[epoch]

    # 사용할 변수들 선언
    correct = 0
    total = 0
    overall_logits = []
    end = time.time()
    running_loss = 0

    
    for iter_, (imgs, labels) in enumerate(iter(loader)): # loader의 인덱스와 데이터를 가져옴
        imgs = imgs.to(device) # imgs를 device로 전송
        labels = labels.to(device) # labels를 device로 전송
        
        outputs = model(imgs) # 모델에 imgs를 넣어서 나온 결과를 outputs에 저장

        criterion = nn.BCEWithLogitsLoss() # 손실함수
        loss = criterion(outputs, labels) # output과 labels을 넣어서 손실 계산
        '''
        have to modify multi label acc
        '''
        outputs = torch.sigmoid(outputs) # simgmoid에 output 집어넣어서 나온 결과를 저장
        # clone() : 텐서를 복제
        # 텐서를 복제해서 사용하면 원본 텐서는 유지한채로, 텐서를 조작할 수 있다
        outputs_preds = outputs.clone() #복제한 텐서를 preds에 저장
        # output : 출력결과
        # .cpu() : cpu로 이동
        # .detach() : 텐서를 그래프에서 분리 (?) > 자동 미분을 방지하기 위해 사용
        # .numpy() : np 배열로 변환
        # .tolist() : 리스트로 변환
        overall_logits += outputs.cpu().detach().numpy().tolist()

        outputs_preds[outputs_preds >= 0.5] = 1 # output_preds의 값이 0.5 이상이면 1
        outputs_preds[outputs_preds < 0.5] = 0 # output_preds의 값이 0.5 미만이면 0

        # print('labels : ',labels)
        # print('outputs_preds :',outputs_preds)
        
        total += labels.size(0) * labels.size(1) # labels.size(0) : 배치 크기, labels.size(1) : 클래스 개수
        correct += torch.sum(outputs_preds == labels.data).item() # 예측값과 실제값이 같다면 correct에 샘플을 누적

        losses.update(loss.item(), imgs[0].size(0)) # losses에 loss.item(), imgs[0].size[0]을 업데이트함
        # loss.item : 현재 loss, img[0].size[0] : 배치 크기
        batch_time.update(time.time() - end) # batch_time에 time.time() - end 업데이트
        # end는 이전에 완료된 시간, time.time()은 현재시간 > 얼마나 걸렸는지 확인하기 위해서
        end = time.time() # end를 현재시간으로 다시 선언

        optimizer.zero_grad() # zero_grad : 모델의 파라미터에 대한 기울기 초기화
        loss.backward() # backwarkd : 손실값에 대해 자동미분 수행
        optimizer.step() # step : 계산된 기울기를 사용해 모델 파라미터 업데이트
        
        running_loss += loss.item() # running_loss에 현재 loss 저장
    
        # iter_ : 현재 반복횟수
        if (iter_ % args.print_freq == 0)& (iter_ != 0): # iter_를 arg.print_frq로 나눈 나머지가 0이고 iter_ 이 0이 아니라면
            # 특정 주기마다 작업을 반복하기 위함
            # progress : utils의 ProgressMeter수행한 결과
            progress.display(iter_) # 현재 iter_의 progress 결과를 출력
            writer.add_scalar('train_loss', running_loss/iter_, (epoch*len(loader))+iter_) # train_loss라는 이름으로 평균손실, 현재 시점 저장
            writer.add_scalar('train_acc', 100.*correct/total, (epoch*len(loader))+iter_) # train_acc라는 이름으로 정확도%, 현재 시점 저장
    
    print('[*] Valid Phase')
    model.eval() # 평가(검증)

    val_batch_time = AverageMeter('Time', ':6.3f') # name은 Time, 형식은 6.3f > 6자리 수, 소숫점 3자리
    val_losses = AverageMeter('Loss', ':.4f') # name은 Loss, 형식은 소숫점 4자리까지
    progress = ProgressMeter( # Utils의 PorgressMeter 실행
        len(loader), # loader의 길이 > 배치수
        [val_batch_time, val_losses], # 모니터링할 지표
        prefix='Epoch: [{}]'.format(epoch)) # Epoch : [epoch]

    val_correct = 0
    val_total = 0
    overall_logits = []
    val_running_loss = 0

    end = time.time()

    with torch.no_grad(): # no_grad() : 기울기 계산하지 않도록 설정
        for iter_, (imgs, labels) in enumerate(iter(val_loader)): # val_loader에서 img랑 labels 가져옴

            # 학습할 환경에서 데이터를 사용하기 위해 하는거..? > cuda를 사용하기 위해서...?
            imgs = imgs.to(device) # imgs를 device로 전송
            labels = labels.to(device) # labels를 device로 전송

            outputs = model(imgs) # model에 imgs 넣어서 나온 출력

            criterion = nn.BCEWithLogitsLoss() # 손실함수
            loss = criterion(outputs, labels) # 손실값 계산
            '''
            have to modify multi label acc
            '''
            outputs = torch.sigmoid(outputs) # 시그모이드 출력값
            outputs_preds = outputs.clone() # 텐서 복제
            overall_logits += outputs.cpu().detach().numpy().tolist() # 결과를 list로 변환해서 누적

            outputs_preds[outputs_preds >= 0.5] = 1 # output이 0.5 이상이면 1
            outputs_preds[outputs_preds < 0.5] = 0 # output이 0.5 미만이면 0
            val_total += labels.size(0) * labels.size(1) # 배치수와 차원수 누적
            val_correct += torch.sum(outputs_preds == labels.data).item() # 예측값과 실제값이 같을 경우 val_correct에 누적
            
            val_losses.update(loss.item(), imgs[0].size(0)) # val_loss에 현재 loss와 배치 크기 업데이트
            val_batch_time.update(time.time() - end) # 현재 시간 - 이전에 완료된 시간 > 얼마나 걸렸는지 확인하기 위함
            end = time.time() # end를 현재시간으로 업데이트
            val_running_loss += loss.item() # val_running_loss에 현재 loss 누적

            if (iter_ % args.print_freq == 0)& (iter_ != 0): # 지정 주기마다 수행하기 위해서 / iter_ == 0일때는 수행안함
                progress.display(iter_) # 현재 반복횟수 반환
                writer.add_scalar('val_loss', val_running_loss/iter_, (epoch*len(val_loader))+iter_) # val_loss라는 이름으로 현재까지의 loss 평균과 현재의 위치? 를 기록
                writer.add_scalar('val_acc', 100.*val_correct/val_total, (epoch*len(val_loader))+iter_) # val_acc라는 이름으로 현재까지의 정확도 %와 위치 기록
        scheduler.step(np.mean(val_running_loss)) # 검증 손실의 평균으로 스케줄러 업데이트
    model.train() # 모델 학습



def test(args, epoch, loader, model, device, writer):

    print('[*] Test Phase')
    model.eval()
    # 모델 평가(검증)
    correct = 0
    total = 0
    overall_logits = []
    overall_preds = []
    overall_gts = []

    with torch.no_grad(): # 기울기 계산 안함
        for iter_, (imgs, labels) in enumerate(iter(loader)): # loader에서 imgs, labels, 인덱스(iter_) 가져옴
            imgs = imgs.to(device) # imgs를 device로 전송
            labels = labels.to(device) # labels를 device로 전송
            outputs = model(imgs) # model에 imgs를 넣어서 나온 결과를 outputs에 저장
            outputs = torch.sigmoid(outputs) # 시그모이드 함수를 사용해서 나온 결과를 outputs에 저장
            outputs_preds = outputs.clone() # 텐서 복제
            overall_logits += outputs.cpu().detach().numpy().tolist() # output을 list 형태로 누적
            outputs_preds[outputs_preds >= 0.5] = 1 # output_preds가 0.5 이상이면 1
            outputs_preds[outputs_preds < 0.5] = 0 # 0.5 미만이면 0
            total += labels.size(0) * labels.size(1) # labels의 배치 수와 클래스 수 저장
            correct += torch.sum(outputs_preds == labels.data).item() # 예측값과 실제 값이 같다면 correct에 누적
        
    test_acc = 100.*correct/total # 정확도 %
    print('[*] Test Acc: {:5f}'.format(test_acc)) # test_acc(정확도)를 소숫점 5자리까지 출력
    writer.add_scalar('Test acc', test_acc, epoch) # Test acc라는 이름으로 정확도와 에포크수를 기록

    model.train() # 모델 학습
    return test_acc # 정확도 반환

def main(args):
    ##### Initial Settings
    json_data = pd.read_json(args.train_path) #json
    class_list = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
                   'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                   'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
                   'Pneumonia', 'Pneumothorax', 'Support Devices'] # warning #json컬럼명 불러오기 [5:] 만큼
    print("[*] class list : " , class_list) # class_list 출력
    num_classes = args.num_class # args의 num_class 저장
    # downstream = '{}_{}_class'.format(args.downstream_name, num_classes) # f문자열 포맷팅 # .format()안의 변수값이 {}에 삽입

    # print('\n[*****] ', downstream) # downstream 출력
    print('[*] using {} bit images'.format(args.bit)) # f문자열 포맷팅 # .format()안의 args.bit값이 {}에 삽입 # [*] using {args.bit} bit images

    # device check & pararrel
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # cuda 사용 가능하다면 cuda를 사용, 아니라면 cpu 사용
    print('[*] device: ', device) # 사용할 device 출력

    # path setting
    pathlib.Path(args.log_dir).mkdir(parents=True, exist_ok=True) 
    pathlib.Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True) # args.checkpoint_dir 경로의 파일 생성
    args.checkpoint_dir = os.path.join(args.checkpoint_dir) 

    # for log
    f = open(os.path.join(args.log_dir,'arguments.txt'), 'w') # args.log_dir/arguments.txt 파일 쓰기 모드로 open
    f.write(str(args)) # args를 문자열로 변환, 변환된 문자열을 arguments.txt에 저장
    f.close() # txt 파일 닫기
    print('[*] log directory: {} '.format(args.log_dir)) # [*] log directory : {args.log_dir}
    print('[*] checkpoint directory: {} '.format(args.checkpoint_dir)) # [*] checkpoint directory: {args.checkpoint_dir}
    
    if args.seed is not None: # args.seed가 None이 아니라면
        random.seed(args.seed) # random.seed를 args.seed로 고정 # 동일한 seed로 고정하여 매번 동일한 결과를 얻을 수 있게 함
        torch.manual_seed(args.seed) # torch.manual_seed를 args.seed로 고정
        # os.environ : 현재 실행중인 파이썬 프로세스들의 환경 변수들을 나타냄
        os.environ['PYTHONHASHSEED'] = str(args.seed) # PYTHONHASHSEED(해시함수 시드)를 arg.seed로 설정
        np.random.seed(args.seed) # np.random.seed를 args.seed로 설정
        torch.cuda.manual_seed(args.seed) # torch.cuda.manual_seed를 args.seed로 설정
        # deterministic : 결정적 연산을 보장
        torch.backends.cudnn.deterministic = True # True로 설정하여 동일한 출력 생성하게 함
        # benchmark : 벤치마킹을 통해 적합한 알고리즘 설정
        torch.backends.cudnn.benchmark = False # False로 설정하여 항상 동일한 알고리즘 사용

    # select network
    print('[*] build network... backbone: {}'.format(args.backbone)) # [*] bulid network... backbon: {args.backbone}
    if args.backbone == 'resnet50': # args.backbone이 'resnet50'이라면
        model = resnet50(num_classes=args.num_class) # model을 resnet50으로 설정
    elif args.backbone == 'vgg': # args.backbone이 vgg라면
        model = vgg16(num_classes=args.num_class) # model을 vgg로 설정
    elif args.backbone == 'densenet': # args.backbone이 'densenet'이라면
        model = densenet169(num_classes=args.num_class) # model을 densenet169로 설정
    elif args.backbone == 'inception': # args.backbone이 'inception'이라면
        model = Inception3(num_classes=args.num_class) # model을 Inception3로 설정
    else:
        ValueError('Have to set the backbone network in [resnet, vgg, densenet]') # backbone network를 누락한 경우, 목록 이외의 값을 설정한 경우 Error

    print(('[i] Total Params: %.2fM'%(calculate_parameters(model)))) # calcuate_parameters(model)을 형식에 맞게 출력
    # % : 형식 지정
    # 2f : 소숫점 두자리
    # M : 백만 ex) 1.23M = 1,230,000
    # SummaryWriter : TensorBoard에 데이터를 기록하기 위한 클래스, 훈련중에 손실, 정확도, 그래프 등 다양한 정보를 기록할 수 있음
    writer = SummaryWriter(args.log_dir) # args.log_dir 경로에 log 파일을 저장

    optimizer = torch.optim.SGD(model.parameters(), # Sigmoid 파라미터 설정
                                lr=args.lr, #learning_rate(학습률)
                                momentum=0.9, # momentum : SGD 변형 / 이전의 90%를 고려해 현재 업데이트에 반영
                                weight_decay=0.) # weight_decay : L2 정규화를 통해 복잡성 줄이는데 사용 / 0 > 가중치 감소 적용 X
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, # ReduceLROnRlateau 스케쥴러 설정
                                         mode='min', # mode : 손실이 최소화되어야함 / min > 관찰하는 값이 최소로 떨어지길 기대, max > 관찰하는 값이 최대가 되길 기대
                                         factor=0.5, # factor : 학습률을 줄이는 비율 50%로 줄이겠다
                                         patience=10,) # patience : 개선이 없는 epoch수, 검증손실 10epoch 동안 개선되지 않으면 학습률 감소

    if args.resume is True: # args.resum이 True라면
        checkpoint = torch.load(args.pretrained) # pretrained의 checkpoint 로드
        pretrained_dict = checkpoint['state_dict'] # checkpoint의 'state_dict' 추출
        pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()} # pretrained_dict의 key, value값들을 돌며 state_dict의 "module"을 ""로 replace
        # load_state_dict : 모델에 학습 가능한 파라미터 로드하는데 사용
        model.load_state_dict(pretrained_dict) # state_dict에 저장된 파라미터를 model에 적용
        args.start_epoch = checkpoint['epoch'] # checkpoint의 epoch 추출
        optimizer.load_state_dict(checkpoint['optimizer']) # checkpoint의 optimizer를 optimizer에 적용
        scheduler.load_state_dict(checkpoint['scheduler']) # checkpoint의 scheduler를 shceduler에 적용
    
    model = model.to(device) # Pytorch모델을 지정된 장치(device)로 이동시킴 (?)
    
    ##### Dataset & Dataloader
    print('[*] prepare datasets & dataloader...')
    # datasets의 DiseaseDataset class
    train_datasets = DiseaseDataset(args.train_path, 'train', args.img_size ,args.bits, args) # DiseaseDataset함수에 args.tain_path, mode = train, img_size, bits, args입력 / 출력은 정규화된 이미지
    val_datasets = DiseaseDataset(args.val_path, 'val', args.img_size, args.bits, args) # mode = val > 출력은 정규화된 이미지
    test_datasets = DiseaseDataset(args.test_path, 'test', args.img_size, args.bits, args) # mode = test > 출력은 정규화된 이미지

    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=args.batch_size, # torch의 Dataloasder로 데이터 불러옴
                                num_workers=args.w, pin_memory=True, # num_works : 데이터 로딩에 사용할 CPU 스레드 수 / pin_memory : 고정된 메모리 영역 사용 (True)
                                shuffle=True, drop_last=True) # shuffle : epoch마다 데이터를 랜덤하게 섞어서 모델에 제공 / drop_last : 전체 데이터를 배치로 나눌때 남는 데이터를 버릴지 > 모든 배치의 크기 일정하게 유지
    val_loader = torch.utils.data.DataLoader(val_datasets, batch_size=args.batch_size,
                                num_workers=args.w, pin_memory=True, drop_last=True) # val데이터는 epoch 돌지 않으므로 선언하지 않음
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=1, # test의 batch_size는 1로 설정 > 각 샘플의 결과를 개별적으로 확인하기 위해? 시간을 측정하기 위해?
                                num_workers=args.w, 
                                pin_memory=True, drop_last=True)
    
    ##### Train & Test
    print('[*] start a train & test loop')
    best_model_path = os.path.join(args.log_dir,'best.pth.tar') # checkpoint_dir에서 best.pth.tar 가져옴

    for epoch in range(args.start_epoch, args.epochs): # start_epoch 부터 epoch 까지 반복 range(a, b)
        train(args, epoch, train_loader, val_loader, model, device, optimizer, writer ,scheduler) # train 수행
        
        '''
        have to modify best loss
        '''
        acc = test(args, epoch, test_loader, model, device, writer) # 정확도 평가 > return 값은 정확도 %
        
        save_name = '{}.pth.tar'.format(epoch) # 저장명 설정 {epoch}.pth.tar
        save_name = os.path.join(args.checkpoint_dir, save_name) # save_name > checkpoint_dir/save_name
        save_model(save_name, epoch, model, optimizer ,scheduler) # model 저장

        if epoch == 0: # 만약 epoch가 0이라면
            best_acc = acc # 최적의 정확도는 acc
            save_model(best_model_path, epoch, model, optimizer , scheduler) # save_model()

        else:
            if best_acc < acc: # best acc가 acc 보다 낮다면
                best_acc = acc # best_acc 갱신
                save_model(best_model_path, epoch, model, optimizer , scheduler) # save_model()

    ##### Evaluation (with best model)
    print("[*] Best Model's Results")
    checkpoint = torch.load(best_model_path) # 베스트모델 로드
    model.load_state_dict(checkpoint['state_dict']) # 베스트 모델의 satae_dict 로드


    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=1, # test_dataset : 테스트에 사용될 데이터셋, batch_size = 1 > 한번에 하나의 샘플만
                                num_workers=args.w, pin_memory=True, drop_last=True) # num_workers : 데이터 로드에 사용할 프로세스 수, pin_memory : 메모리 고정(True), drop_last : batch를 돌고 남은 데이터 버릴지

    # evaluate(args, test_loader, model, device, num_classes, class_list) # 평가
    
if __name__ == '__main__':
    argv = parse_arguments(sys.argv[1:]) # 0번을 제외한 모든 인수를 가져옴
    main(argv) # argv => args




