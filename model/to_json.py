import os
import sys
import cv2
import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from tqdm import tqdm
import json
import argparse


"""
실행방식 ex)
!python to_json.py \
    --csv_path ../main_combined.csv \
    --output_dir ./outptu \
    --split_ratios 0.7 0.2 0.1 \
    --type uni
"""

def make_json(csv_path, type, split_ratios):
    # CSV 파일 읽기
    csv_data = pd.read_csv(csv_path)

    # type
    # all : 전체 데이터 -> if 문 실행 x
    # uni : subject id 별 하나의 이미지만
    if type == 'uni':
        data = csv_data.drop_duplicates(['subject_id'])
    else:
        data = csv_data

    # 결과 저장을 위한 딕셔너리 초기화
    res_dict = {'imgs': [], 'labels': [], 'set': []}

    # 이미지 경로 및 레이블 데이터 정의
    X = data['PNGPath'].values
    y = data.iloc[:, 4:-1].values

    # y 값이 1이면 1, 아닌 것은 0으로 치환
    y_modified = np.where(y == 1, 1, 0)

    # train:test+val 비율 설정
    train_ratio = split_ratios[0]
    val_ratio = split_ratios[1] / (split_ratios[1] + split_ratios[2])

    # MultilabelStratifiedKFold 설정
    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError("Train 비율은 0보다 크고 1보다 작아야 합니다.")

    mskf = MultilabelStratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # train과 나머지(temp) 인덱스 나누기
    train_idx, temp_idx = next(mskf.split(X, y_modified))

    # train 세트와 나머지 세트 나누기
    X_train, X_temp = X[train_idx], X[temp_idx]
    y_train, y_temp = y_modified[train_idx], y_modified[temp_idx]

    # 나머지 세트를 validation과 test로 나누기
    if val_ratio <= 0 or val_ratio >= 1:
        raise ValueError("Validation 비율은 0보다 크고 1보다 작아야 합니다.")

    mskf_temp = MultilabelStratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    val_idx, test_idx = next(mskf_temp.split(X_temp, y_temp))

    # 각 세트를 할당된 인덱스에 따라 res_dict에 추가
    for idx in train_idx:
        res_dict["imgs"].append(f".{X[idx]}")  # 경로 앞에 . 추가
        res_dict["labels"].append(y_modified[idx].tolist())
        res_dict["set"].append("0")

    for idx in val_idx:
        res_dict["imgs"].append(f".{X_temp[idx]}")  # 경로 앞에 . 추가
        res_dict["labels"].append(y_temp[idx].tolist())
        res_dict["set"].append("1")

    for idx in test_idx:
        res_dict["imgs"].append(f".{X_temp[idx]}")  # 경로 앞에 . 추가
        res_dict["labels"].append(y_temp[idx].tolist())
        res_dict["set"].append("2")

    return res_dict


def split_json(output_dir, json_data, set_name):
    file_name = f'{set_name}.json'
    os.makedirs(output_dir, exist_ok=True)  # 디렉토리가 없으면 생성
    output_path = os.path.join(output_dir, file_name)
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=4)

    print(f"\n{set_name} 데이터: {len(json_data['imgs'])}개")
    print(f"{file_name} 파일이 {output_path}에 저장되었습니다.")


def calculate_distribution(dataset, set_name):
    labels = np.array(dataset['labels'])
    label_counts = np.sum(labels, axis=0)
    total_samples = len(labels)
    print(f"\n{set_name} 데이터 분포:")
    print(f"총 샘플 수: {total_samples}")
    for i, count in enumerate(label_counts):
        print(f"레이블 {i}: {count}개 ({(count / total_samples) * 100:.2f}%)")


def main():
    # argparse 설정
    parser = argparse.ArgumentParser(description="JSON 데이터 분할 및 저장 스크립트")
    parser.add_argument('--csv_path', type=str, required=True, help="원본 CSV 파일 경로")
    parser.add_argument('--output_dir', type=str, required=True, help="JSON 파일 저장 경로")
    parser.add_argument('--split_ratios', type=float, nargs=3, default=[0.7, 0.2, 0.1],
                        help="Train, Validation, Test 비율 (기본값: 0.7 0.2 0.1)")
    parser.add_argument('--type', type=str, choices=['all', 'uni'], required=True,
                        help="데이터 타입 (all: 전체, uni: subject_id별 하나)")

    args = parser.parse_args()

    # JSON 데이터 생성
    train_dict = make_json(args.csv_path, args.type, args.split_ratios)

    # JSON 파일명 설정
    prefix = 'uni_' if args.type == 'uni' else ''
    json_name = f"{prefix}dataset.json"
    output_path = os.path.join(args.output_dir, json_name)

    # JSON 파일 저장
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(train_dict, f, indent=4)

    print(f"데이터셋 분할 완료! JSON 파일이 {output_path}에 저장되었습니다.")

    # JSON 파일 불러오기
    with open(output_path, 'r') as f:
        data = json.load(f)

    # 세트별 데이터 분포 확인 및 저장
    train_data = {'imgs': [], 'labels': []}
    valid_data = {'imgs': [], 'labels': []}
    test_data = {'imgs': [], 'labels': []}

    for img, label, set_type in zip(data['imgs'], data['labels'], data['set']):
        if set_type == "0":
            train_data['imgs'].append(img)
            train_data['labels'].append(label)
        elif set_type == "1":
            valid_data['imgs'].append(img)
            valid_data['labels'].append(label)
        elif set_type == "2":
            test_data['imgs'].append(img)
            test_data['labels'].append(label)

    calculate_distribution(train_data, "Train")
    calculate_distribution(valid_data, "Validation")
    calculate_distribution(test_data, "Test")

    split_json(args.output_dir, train_data, "train")
    split_json(args.output_dir, valid_data, "val")
    split_json(args.output_dir, test_data, "test")


if __name__ == '__main__':
    main()
