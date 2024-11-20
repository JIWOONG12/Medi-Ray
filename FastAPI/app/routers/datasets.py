import os
import sys
import numpy as np
import torch
import random
import glob

from torch.utils.data import Dataset

import torchvision
# import torchvision.transforms as T

from config import parse_arguments
from PIL import Image
import cv2
import SimpleITK as sitk
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DiseaseDataset(Dataset):
    def __init__(self, input_path, mode, image_size, bits, args, transform=None):
        # def __init__ (self, ...) : 클래스의 생성자, 객체가 생성될 때 자동으로 호출
        # input_path : 입력데이터의 경로
        # mode : 데이터 셋의 모드 ex) train, test, valid > 데이터셋의 용도에 따라 다르게 설정
        # image_size : 이미지 크기
        # bits : 이미지의 색상 깊이
        # args : 하이퍼파라미터 및 설정 포함하는 객체
        # transform : 변환 옵션, None로 설정 > 변환 지정할 수 있음
        self.mode = mode # Unused variable. However, it will be used for transform
        self.args = args
        self.image_size = image_size
        self.bits = bits
        with open(input_path, "r") as f: # input_path의 파일을 read 형식으로 open해서 f로 저장
            self.samples = json.load(f) # json파일에서 f로드하여 self.sample에 저장
        
        """
        augmentation strategy 
        """

        if mode == 'train': # 만약 mode가 train이라면
            if args.aug == True: # args.aug가 True라면 / aug : 데이터 증강(data augmentation) 사용 여부
                self.transform = A.Compose([ # Compose() 는 여러개의 이미지 변환을 연속적으로 적용하기 위해 사용
                    A.Resize(self.image_size, self.image_size), # 이미지 크기 조정
                    A.OneOf([ # 랜덤으로 하나 선택하여 적용
                        A.MedianBlur(blur_limit=3, p=0.1), # 최대 3의 블러를 적용하는 MemianBlur를 10%확률로 적용
                        A.MotionBlur(p=0.2), # MotionBlur를 20% 확률로 적용
                        # A.IAASharpen(p=0.2), # IAASharpen을 20%확률로 적용
                        ], p=0.2), # 전체를 20%확률로 적용
                    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=10, p=0.2), # 이미지 이동, 크기조절, 회전 / limit 지정되어있음 20% 확률로 랜덤하게 적용
                    A.OneOf([
                        # OpticalDistortion : 왜곡, 굴곡, 외삽
                        # A.OpticalDistortion(p=0.3), # OpticalDistortion이 30% 확률로 적용
                        ], p=0.2), # 전체가 20% 확률로 적용
                    # A.OneOf([
                    #     A.CLAHE(clip_limit=4.0),
                    #     A.Equalize(),
                    #     ], p=0.2),
                    A.OneOf([
                        A.GaussNoise(p=0.2), # 가우스 노이즈를 20% 확률로 적용
                        A.MultiplicativeNoise(p=0.2), # MultiplicativeNoise를 20%확률로 적용
                        ], p=0.2), # 전체를 20% 확률로 적용
                    # HueSaturationVlaue : 색조, 채도, 밝기를 변경
                    # A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=0, val_shift_limit=0.1, p=0.3), # 색조 limit : 0, 채도 limit : 0, 밝기 limit : 0.1, 30% 확률로 적용
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    # A.Normalize(mean=(0.485), std=(0.229)), # 정규화 / 평균 : 0.485, 표준편차 : 0.229
                    ToTensorV2(), # PyTorch 텐서로 변환
                    ])
            else: # args.aug가 False라면
                self.transform = A.Compose([
                    A.Resize(self.image_size, self.image_size),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    # A.Normalize(mean=(0.485), std=(0.229)), # 정규화 / 평균 : 0.485, 표준편차 : 0.229
                    ToTensorV2(),
                    ])
        else: # mode가 train이 아니라면
            self.transform = A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                # A.Normalize(mean=(0.485), std=(0.229)), # 정규화 / 평균 : 0.485, 표준편자 : 0.229
                ToTensorV2(),
                ])

    def __getitem__(self, idx): # 함수 getitem 선언
        # processing > 밑에 선언되어있음
        # transform : Albumentations 라이브러리의 변환은 적용하는 객체
        # transform(image=...) > ['image'] 를 통해 변환된 이미지 얻음
        imgs = self.transform(image=self._preprocessing(self.samples['imgs'][idx] , # samples : 특정 idx에 해당하는 imgs 추출, prepocessing > 전처리
                                self.bits))['image'] # preprocessing > bits 조정
        labels = np.array(self.samples['labels'][idx]).astype(np.float32) # samles에서 idx에 해당하는 레이블 데이터 추출, float32 형식의 np.array 로 변환
        return imgs, labels
            
    def __len__(self):
        return len(self.samples['labels']) # labels의 갯수 반환
    
    def _preprocessing(self, path , bits):
        if bits == 8: # bits가 8이라면
            img = np.array(Image.open(path)) # path의 이미지를 배열 형식으로 저장
            # img = self._min_max_scaling(img) # 불러온 이미지에 대해 min_max_scaling() 적용
        else:
            img = np.load(path) # bit가 8이 아니라면
        # img = self._standardization(img) # standardization() 적용
        return img

    # def _min_max_scaling(self, img):
    #     # min_max 정규화
    #     return (img-np.min(img)) / (np.max(img)-np.min(img)) # (각 픽셀값 - 최소 픽셀값) / (최대 픽셀 값 - 최소 픽셀 값) > 0~1 정규화
    #     # np.min() > 이미지 최소 픽셀 값, np.max() > 이미지 최대 픽셀값
    
    # def _standardization(self, img):
    #     np_img = img.astype(np.float32) # img를 float32 형식으로 변환
    #     mean, std = np.mean(np_img), np.std(np_img) # 평균 : img의 평균값, std : 이미지의 표준편차
    #     norm_img = (np_img-mean)/std # 정규화 수행
    #     # (각 픽셀값 - 평균값) / (표준편차)
    #     return norm_img
# return값은 정규화된 이미지☆

# For test
if __name__ == '__main__':
    dataset = DiseaseDataset('./json/dummy.json') # DiseaseDataset 클래스 인스턴스 생성. json 파일에서 데이터 로드하여 dataset에 저장
    # torch.utils.data.DataLoader() : 데이터 셋을 배치 단위로 로드
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1) # 배치사이즈 = 1, num_sorkers : 데이터 로드하기 위해 사용할 서브프로세스의 수

    for imgs, labels in iter(dataloader): # dataloader에서 imgs : 이미지데이터, labels : 레이블 가져옴 / iter사용하여 dataloader를 반복 가능한 객체로 만듦
        print(imgs.shape, labels.shape) # 이비지와 레이블의 shape를 출력
        
