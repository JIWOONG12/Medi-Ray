import sys
sys.path.append('./MONAI')
from monai.visualize import GradCAM
import random

from config import parse_arguments
from datasets import DiseaseDataset
from models.resnet import resnet50
from models import vgg16
import pandas as pd
from utils_folder.eval_metric import *

import torch

import pathlib
import cv2

# 람다함수 : 익명함수, 함수의 이름을 명시하지 않고 한 줄로 간단히 정의 할 수 있는 Python 함수
# lambda 인자들 : 반환값
fn_tonumpy = lambda x : x.to('cpu').detach().numpy().transpose(0,2,3,1)
# x.to('cpu') : 텐서를 cpu로 보냄
# detach() : 텐서의 연산 그래프에서 해당 텐서를 분리하는 역할을 함, 학습과정에서 역전파를 수행하지 않도록 함
# numpy() : 넘파이 배열로 변환
# transpose(0,2,3,1) : 배열 차원순서 변경
# 파이토치 텐서는 저장 순서 : (batch, channel, height, width)
# 넘파이 배열 순서 : (batch, height, width, channel)


""" 왜 그렇게 생각했는지 설명하기 위해 사용(?) """


def evaluate_cam(args, loader, model, device, num_classes , class_list, args.log_dir):

    model.eval() # 평가
    save_dir = args.log_dir # log_dir : 로그 저장

    for iter_, (imgs, labels) in enumerate(iter(loader)): # loader에서 iter_, imgs, labels 가져옴

        imgs = imgs.to(device) # imgs를 device로 전송
        labels = labels.to(device, dtype=torch.long) # labels를 device로 전송, data type은 64비트 정수형

        outputs = model(imgs) # img를 model에 넣어서 나온 결과를 output에 저장
        # GradCAM 객체 생성
        cam = GradCAM(nn_module = model, target_layers = 'layer4') # nn_module : 사용할 모델 지정, target_layers : 모델의 레이어 지정, 보통 마지막 합성곱 층을 지정
        result = cam(x=imgs, layer_idx=-1) # cam 실행, 활성화맵 계산, layer_idx=-1 : 지정된 레이어의 마지막 활성화 맵 사용
        result = result.squeeze() # squeeze() : 차원이 1인 축을 제거, 불필요한 차원 없애고 2D 이미지로 변환
        heatmap = np.uint8(255 * result) # 0~255 사이 값으로 변환, 히트맵으로 만든다  #unit8 : 8비트 정수형
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) # applyColerMap : 히트맵에 컬러 적용
        # COLORMAP_JET : 파란색에서 빨간색으로 변하는 컬러맵,  중요한 부분 빨간색, 덜 중요한 부분 파란색
        heatmap = heatmap/255 # 히트맵 값 정규화 0~1

        gt_imgs = fn_tonumpy(imgs) # imgs > np 배열로 변환
        # np.stack() : 여러 배열을 새로운 축을 기준으로 쌓아줌
        cam_imgs = np.stack((gt_imgs[0].squeeze(),)*3, axis=-1) # gt_imgs[0] : 첫번째 이미지, squeeze() : 데이터 2D로 만듦 > *3 3번 반복해서 3차원으로 변환, axis=-1 : 가장 마지막 축에 배열을 쌓음
    
        superimposed_img = heatmap*0.2 + cam_imgs #히트맵 값에 0.2 곱해서 강도 낮춤, + cam_imgs : 원본이미지와 히트맵 결합
        superimposed_img /= 1.2 # 이미지의 값들을 1.2로 나눔, 밝기를 조절

        full_image = np.zeros((512,1024,3)) # (512, 1024, 3)의 이미지 생성
        full_image[:512, :512, :] = gt_imgs[0][:,:,0]*255
        # full_image의 오른쪽 절반을 gt_img[0][:,:,0]*255로 채움, gt_img[0] : 원본이미지, 255 곱해서 0~255 사이 값으로 변환
        full_image[:512, 512:1024, :] = superimposed_img*255
        # full_image의 왼쪽 절반을 superimposed_img*255로 채움, superimposed_img : 히트맵과 결합한 이미지

        cv2.imwrite(os.path.join(save_dir,'{}_cam.jpg'.format(iter_)), full_image) #이미지 저장
        # full_image를 save_dir/{iter_}_cam.jpg로 저장

def main(args):
    ##### Initial Settings
    csv_data = pd.read_csv(args.csv_file) # csv 파일 불러옴
    class_list = csv_data.keys().tolist()[5:] # warning # class_list : csv_data.keys()의 5번째 데이터 부터 리스트로 저장
    print("[*] class list : " , class_list)
    num_classes = args.num_class # num_classes에 args.num_class를 저장
    downstream = '{}_{}_class'.format(args.downstream_name, num_classes) # 문자열 생성, downstream 이름 지정

    print('\n[*****] ', downstream)
    print('[*] using {} bit images'.format(args.bit))

    # device check & pararrel
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # cuda가 사용가능하다면 device = cuda, 아니라면 cpu
    print('[*] device: ', device) # device명 출력

    # path setting
    pathlib.Path(args.log_dir).mkdir(parents=True, exist_ok=True) # log_dir 폴더 생성
    pathlib.Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True) # chenckpoint_dir 폴더 생성

    folder_name = '{}_{}_{}_bit'.format(args.message, downstream, args.bit) # 폴더 이름 지정

    args.log_dir = os.path.join(args.log_dir, folder_name) # log_path에 folder_name 추가해서 갱신
    pathlib.Path(args.log_dir).mkdir(parents=True, exist_ok=True) # arg.log_dir 파일 생성
    
    # for log
    f = open(os.path.join(args.log_dir,'arguments.txt'), 'w') # argumentx.txt 파일 쓰기 모드로 open
    f.write(str(args)) # args를 문자열로 변환해서 f에 저장
    f.close() # 파일 닫기
    print('[*] log directory: {} '.format(args.log_dir))
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed) # os 자체의 seed 고정
        np.random.seed(args.seed) # numpy seed 고정 
        torch.cuda.manual_seed(args.seed) # cudnn seed 고정
        torch.backends.cudnn.deterministic = True # cudnn seed 고정(nn.Conv2d)
        torch.backends.cudnn.benchmark = False # CUDA 내부 연산에서 가장 빠른 알고리즘을 찾아 수행


    # select network
    print('[*] build network... backbone: {}'.format(args.backbone))
    if args.backbone == 'resnet50': # backbone 모델 결정
        model = resnet50(num_classes=args.num_class)
    elif args.backbone == 'vgg':
        model = vgg16(num_classes=args.num_class)
    elif args.backbone == 'densenet':
        model = densenet169(num_classes=args.num_class)
    elif args.backbone == 'inception':
        model = Inception3(num_classes=args.num_class)
    else:
        ValueError('Have to set the backbone network in [resnet, vgg, densenet]')

    model = model.to(device) # 모델을 device에 전송

    if args.resume is True: # resume이 True라면
        checkpoint = torch.load(args.pretrained) # checkpoint에 pretrained 모델 불러옴
        pretrained_dict = checkpoint['state_dict'] # checkpoint의 'state_dict' 추출
        pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()} # pretrained_dict의 key, value값들을 돌며 state_dict의 "module"을 ""로 replace
        # load_state_dict : 모델에 학습 가능한 파라미터 로드하는데 사용
        model.load_state_dict(pretrained_dict) # pretrained_dict의 파라미터 model에 불러옴
        print("load model completed")
    else:
        ValueError('Have to input a pretrained network path')

    ##### Dataset & Dataloader
    print('[*] prepare datasets & dataloader...')
    test_datasets = DiseaseDataset(args.test_path, 'test', args.img_size, args.bits, args) # DiseaseDataset함수에 args.test_path, mode = test, img_size, bits, args입력 / 출력은 정규화된 이미지
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=1, # batch_size : 이미지마다의 결과를 확인하기 위해 1로 설정
    num_workers=args.w, pin_memory=True, drop_last=True) # now_workers : 스레드 수 지정, pin_memory : 고정된 메모리 영역 할당, drop_last : 배치로 나누고 남은 데이터 버림
    
    ##### Train & Test
    print('[*] start a test')
    evaluate_cam(args, test_loader, model, device, num_classes , class_list , args.log_dir) #evaluate_cam 실행 > 히트맵 이미지 출력
        
if __name__ == '__main__':
    argv = parse_arguments(sys.argv[1:])
    main(argv)
