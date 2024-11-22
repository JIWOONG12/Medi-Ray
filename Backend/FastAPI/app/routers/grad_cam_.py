import sys
sys.path.append('./MONAI')
from models.resnet import resnet50
import torch.nn as nn
import torchvision; torchvision.version
import torchvision.models as models
import os
import torch
from monai.visualize import GradCAM
from PIL import Image
import pathlib
import numpy as np

from app.func import gradcam


def main():
    ##### 초기 설정

    filename = '/adamw_weight_updated_imagnet_uni_224_bs128_1e-4'
    class Args:
        def __init__(self):
            self.num_class = 14  # 클래스 수
            self.backbone = 'resnet50'  # 백본 모델 선택
            self.log_dir = './runs' + filename  # 로그 디렉토리
            self.img_size = 224  # 이미지 크기
            self.bits = 8  # 비트 수
            self.seed = 50  # 랜덤 시드
            self.resume = True  # 재개 여부
            self.pretrained = './runs' + filename + '/best.pth.tar'  # 프리트레인 모델 경로

    args = Args()

    # 사용자로부터 이미지 경로 입력 받기
    img_path = input("Enter the path to the image file: ").strip()
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"[ERROR] The provided image path does not exist: {img_path}")

    print(f"선택된 이미지 경로: {img_path}")

    # 클래스 리스트 정의
    class_list = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
                   'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                   'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
                   'Pneumonia', 'Pneumothorax', 'Support Devices']

    # 디바이스 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('[*] device: ', device)

    # 로그 디렉토리 생성
    pathlib.Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # 랜덤 시드 설정
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 모델 선택 및 로드
    print('[*] 네트워크 구축 중... 백본: {}'.format(args.backbone))
    if args.backbone == 'resnet50':
        model = resnet50(num_classes=args.num_class)
    elif args.backbone == 'vit':
        vit = models.vit_b_16(pretrained=True)
        head_in_features = vit.heads.head.in_features
        vit.heads.head = nn.Linear(head_in_features, args.num_class)
        model = vit
    else:
        raise ValueError('백본 네트워크는 [resnet, vgg, densenet] 중 하나여야 합니다.')

    model = model.to(device)

    if args.resume:
        checkpoint = torch.load(args.pretrained)
        pretrained_dict = checkpoint['state_dict']
        pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
        model.load_state_dict(pretrained_dict)
        print("모델 로드 완료")
    else:
        raise ValueError('프리트레인 네트워크 경로를 입력해야 합니다.')

    ##### GradCAM 평가 시작
    print('[*] GradCAM 평가 시작')
    gradcam(args, img_path, model, device, args.num_class, class_list, args.log_dir)

