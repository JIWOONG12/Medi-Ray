import os
import sys
import numpy as np
import random
import glob22
import natsort
import cv2
import tqdm
import random
import pandas as pd
import json


def make_dict(csv_path , whole_img_list):
    
    data = pd.read_csv(csv_path) # data 불러오기
    res_dict = {'imgs':[], 'labels':[]} # res_dict는 imgs랑 labels로 이루어짐

    for i in tqdm.tqdm(range(len(data))): # data의 길이만큼 반복
        row = data.iloc[i] # 행은 data[i]
        if len(row) < 14: # low의 개수가 14보다 작으면
            labels = [0] * 14 # labels는 14개의 0으로 채운 리스트
        else: # 14 이상이라면
            labels = [] # labels 리스트 선언
            for col in row[5:]: # 행의 5번째부터
                if col == 1 or col == -1: # 열 값이 1이거나, -1이면
                    labels.append(1) # labels에 1 추가
                else: # 아니라면
                    labels.append(0) # labels에 0추가
        # 이미지 이름 : data의 Path열에서 데이터 추출
        # data.iloc[i]['Path'] > 경로이기 때문에 a/b/c/d.png 이런식
        # '/'로 스플릿 하고 [-3], [-2], [-1] > 뒤에서부터 3, 2, 1 번째 데이터 가져옴
        # 이미지 명은 d.png 이런식이기에 '.'기준으로 스플릿 하고 0번 데이터 가져옴
        img_name = data.iloc[i]['Path'].split('/')[-3] +  + data.iloc[i]['Path'].split('/')[-2] + '_' + data.iloc[i]['Path'].split('/')[-1].split('.')[0] + '.png'
        matching_img_path = [s for s in whole_img_list if img_name in s] # 리스트내포
        # whole_img_list의 항목 s 돌면서 img_name이 포함된 경우에만 리스트에 추가
        
        try:
            res_dict['imgs'].append(matching_img_path[0]) # img경로가 있으면 res_dict의 이미지에 추가하고
            res_dict['labels'].append((labels)) # labels도 추가
        except:
            print('[*] Error file ' , matching_img_path , img_name ) # 없으면 error 출력
    return res_dict


def main():

    open_dataset = 'CheXpert-v1.0' # 데이터셋 지정
    phase = 'valid' # 검증용
    root_dir = '/mnt/nas107/open_dataset' # 기본 경로
    data_dtype = 'png' # 데이터 타입 png
    img_size = '512' # 이미지 사이즈

    csv_path = '{}/{}/{}.csv'.format(root_dir, open_dataset, phase) # csv 파일 경로

    img_folder_path_ = natsort.natsorted(glob.glob('{}/{}/preprocessed/{}/{}/{}/*'.format(root_dir, open_dataset, data_dtype, phase, img_size )))
    # 이미지파일 경로 생성
    # glob : 경로를 리스트로 반환 > root_dir의 open_dataset의 data_dtype이 png인, 512*512 크기의 valid 이미지 가져옴
    # natsort : 파일 이름 정렬
    
    whole_img_list = [] # 리스트 선언

    for i in tqdm.tqdm(range(len(img_folder_path_))): # img_foler_path_의 길이만큼 수행
        in_dir_img_list = natsort.natsorted(glob.glob(img_folder_path_[i] + '/*.png')) # 파일 경로에 있는 png 파일 리스트로 저장
        for idx in range(len(in_dir_img_list)): # in_dir_img_list 만큼 수행
            whole_img_list.append(in_dir_img_list[idx]) # whole_img_list에 인덱스 추가

    train_dict = make_dict(csv_path , whole_img_list) # csv_path에 인덱스 리스트 추가해서 딕셔너리 생성
    json_name = './json/{}_chexPert_16_{}.json'.format(phase, img_size) # json파일 이름 설정 > ./json/{phase}_chexPert_16_{img_size}.json
    with open(json_name, 'w') as f: # json 파일 open
        json.dump(train_dict, f, indent=4) # train_dict를 json 형식으로 변환해서 저장, indent > 들여쓰기 4칸으로 지정

if __name__ == '__main__':
    main()
