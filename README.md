## 👨‍⚕️ Medi-Ray : 핵심프로젝트

![image](https://github.com/user-attachments/assets/a1229fd2-0fa0-430a-a6d8-acafea361835)

## 📑 프로젝트 소개
### 주제 : Vision Transformer를 활용한 흉부 방사선 폐 진단 서비스
 인공지능사관학교 사물지능 과정 핵심프로젝트로 기업의 멘토링을 받아 개발하였습니다.
  - **ViT(비전 트랜스포머) 모델을 활용**하여 높은 효율성 기대
  - **Dicom 형식의 대규모 공개 데이터 셋
    (MIMIC-CXR Database v2.1.0 4.7TB)을 활용**함으로 데이터 투명성 제공
  - **Multi-Label Classification**을 통한 최대 14가지 비정상 소견을 진단 및 보조
  - 의사의 편의성 향상을 위해 흉부 X-ray에 직접 **붓그림이나 사각형을 그릴 수 있는 도구 기능**

## 👨‍👦‍👦 팀원 소개
| 이지웅 | 최문경 | 윤주향 | 박주형 |
|:---:|:---:|:---:|:---:|
| <img src="https://github.com/user-attachments/assets/8169f93c-1e45-473f-84c3-e07a1a09069b" width="150" /> | <img src="https://github.com/user-attachments/assets/99ec626b-50a8-4a58-9eba-331f5d45f4e3" width="150" /> | <img src="https://github.com/user-attachments/assets/e8060276-734a-47a6-a8a1-e261730cff7d" width="150" /> | <img src="https://github.com/user-attachments/assets/a91ad806-d42a-4521-8a10-174371efafcc" width="150" /> |
| [@JIWOONG12](https://github.com/JIWOONG12) | [@mooonkyeong](https://github.com/mooonkyeong) | [@JuHyang-Y](https://github.com/JuHyang-Y) | [@JuHyang-Y](https://github.com/JuHyang-Y) |
| PM & Modeling |  Modeling | Back-End & DB | Front-end |


## 🎥 시연 영상
https://github.com/user-attachments/assets/ac672086-a89f-4133-89e5-b7196d88d683


## 📅 프로젝트 기간
* **계획 / 분석 / 설계** : ２４. ０９. ２３ ~ ２４. １０. １８
* 　　　**구현**　　　 : ２４. １０. １８ ~ ２４. １１. ２５
 
## 🔨 사용 기술
  #### ✔️Back-end
<img src="https://img.shields.io/badge/java-E34F26?style=for-the-badge&logo=java&logoColor=white"><img src="https://img.shields.io/badge/springboot-6DB33F?style=for-the-badge&logo=springboot&logoColor=white"><img src="https://img.shields.io/badge/fastapi-009688?style=for-the-badge&logo=python&logoColor=white"><img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=yellow">
  #### ✔️front-end
<img src="https://img.shields.io/badge/html5-E34F26F?style=for-the-badge&logo=html5&logoColor=green"><img src="https://img.shields.io/badge/css3-1572B6?style=for-the-badge&logo=css3&logoColor=yellow"><img src="https://img.shields.io/badge/javascript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=yellow"><img src="https://img.shields.io/badge/tailwindcss-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=yellow">
  #### ✔️Model
<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=yellow"><img src="https://img.shields.io/badge/pytorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=yellow">
  #### ✔️Database
<img src="https://img.shields.io/badge/mariadb-003545?style=for-the-badge&logo=mariadb&logoColor=yellow">

## 🗃 E-R Diagram
![image](https://github.com/user-attachments/assets/bf7535ec-76c5-40bd-8bdf-17f9cd31f0de)

## ✒ DataSet

 ### 데이터 출처
  #### 📊 MIMIC_CXR (4.7TB)
- MIT 계산생명연구실에서 관리하는 의학 연구 데이터 저장소
- 법수 교육이나 선행연구에 위해 교육 수료증과 같은 승인 절차 필요
- 총 227,835건의 환자사진 연구에 해당하는 377,110개의 X-ray 이미지
- 4.7TB DICOM 형식의 파일

  #### 📋 DICOM (Digital Imaging and Communications in Medicine)
<img src="https://github.com/user-attachments/assets/52f6447e-418e-4e86-bd40-fb938e6e87f6" width="50%" alt="DICOM Logo">

- 국제 의료 디지털 영상 표준 형태
- X-ray, CT, MRI와 같은 전세계 모든 의료 영상은 DICOM형식으로 저장
- 고해상도와 다차원으로 매우 큰 용량
- 환자의 나이, 성별 등의 다양한 메타데이터를 포함
- 의사 소견서와 함께 제공되어 상세한 진단 정보 확인 가능

 ## 1. 데이터 전처리
  ### 🖼️ Histogram Equalization
- 화질 개선을 목적으로 영상의 밝기 분포를 재분배하여 영상의 밝기를 균일하게 만들어주는 기술
- 원본 이미지와 처리된 이미지의 비교를 통해 선명도 향상 확인 가능
![image](https://github.com/user-attachments/assets/ee688690-474e-41e8-9a3d-ae3858f2618f)

 ## 2. MIMIC-CXR-2.0.0-negbio 라벨 데이터셋
  ### 📈 데이터셋 특징
- 다양한 촬영 방식(PA, AP 등)과 상태로 구성
- 총 14개의 결려에 대해 1, 0, -1의 값을 가짐
- 하나의 Subject_id(연구번호)에 대해 여러개의 이미지를 가짐
- 총 36,040개의 이미지 데이터

  ### 📊 레이블 분포표
| 번호 | Label | 개수 (비율) |
|------|--------|------------|
| 1 | Atelectasis | 4685개 (13.97%) |
| 2 | Cardiomegaly | 4126개 (11.45%) |
| 3 | Consolidation | 905개 (2.51%) |
| 4 | Edema | 2197개 (6.10%) |
| 5 | Enlarged | 731개 (2.03%) |
| 6 | Fracture | 621개 (1.72%) |
| 7 | Lung Lesion | 932개 (2.59%) |
| 8 | Lung Opacity | 5616개 (15.58%) |
| 9 | No Finding | 18829개 (52.24%) |
| 10 | Pleural Effusion | 4161개 (11.55%) |
| 11 | Pleural Other | 252개 (0.70%) |
| 12 | Pneumonia | 1958개 (5.43%) |
| 13 | Pneumothorax | 560개 (1.55%) |
| 14 | Support Devices | 5162개 (14.32%) |
| 총 데이터 수 | | 36,040개 |

 ## 3. MultilabelStratifiedKFold 데이터 분할
  ### 🔄 분할 방식
- 다중 레이블(Multi-label) 데이터의 불균형을 고려하여 데이터를 K개의 폴드로 나누는 교차검증 기법
- Multi-label 간의 상관 관계를 유지하고 희소 데이터를 확보하는 방식 채택

### 📊 데이터 분할 상세
#### Train/Validation/Test 데이터 분포

| 레이블 | Train (24,027개) | Validation (8,008개) | Test (4,005개) |
|--------|------------------|---------------------|----------------|
| 레이블 0 | 3124개 (13.00%) | 1040개 (12.99%) | 521개 (13.01%) |
| 레이블 1 | 2751개 (11.45%) | 916개 (11.44%) | 459개 (11.46%) |
| 레이블 2 | 603개 (2.51%) | 202개 (2.52%) | 100개 (2.50%) |
| 레이블 3 | 1465개 (6.10%) | 488개 (6.09%) | 244개 (6.09%) |
| 레이블 4 | 488개 (2.03%) | 162개 (2.02%) | 81개 (2.02%) |
| 레이블 5 | 414개 (1.72%) | 138개 (1.72%) | 69개 (1.72%) |
| 레이블 6 | 622개 (2.59%) | 207개 (2.58%) | 103개 (2.57%) |
| 레이블 7 | 3744개 (15.58%) | 1248개 (15.58%) | 624개 (15.58%) |
| 레이블 8 | 12553개 (52.25%) | 4184개 (52.25%) | 2092개 (52.23%) |
| 레이블 9 | 2774개 (11.55%) | 924개 (11.54%) | 463개 (11.56%) |
| 레이블 10 | 168개 (0.70%) | 56개 (0.70%) | 28개 (0.70%) |
| 레이블 11 | 1305개 (5.43%) | 435개 (5.43%) | 218개 (5.44%) |
| 레이블 12 | 373개 (1.55%) | 125개 (1.56%) | 62개 (1.55%) |
| 레이블 13 | 3442개 (14.33%) | 1146개 (14.31%) | 574개 (14.33%) |
| **총계** | **24,027개** | **8,008개** | **4,005개** |

---

## 📚 주요 기능
### 1. Explainable AI 
#### - Grad CAM
- 모델이 이미지의 어느 부분을 주목했는지 히트맵(Heat-map)으로 시각화
- AI의 판단 근거를 직관적으로 이해할 수 있게 함

<p align="center">
<img src="https://github.com/user-attachments/assets/d715e4d4-5fef-41bb-88a9-3940178c6137" width="80%" />
</p>

### 2. Canvas API 기반 드로잉 기능
- Javascript와 HTML canvas 엘리먼트를 통해 그래픽을 그릴 수 있는 수단을 제공하는 API
- 진단 결과를 더 쉽고 직관적으로 설명할 수 있도록 하기 위함
- 주요 기능: 자유 곡선 그리기, 사각형 그리기, 그리기 도구 설정(색상, 선 굵기), 캔버스 초기화

<p align="center">
<img src="https://github.com/user-attachments/assets/70bd4d0c-af9f-42a8-a723-9779373b1854" width="30%" />
<br>
<em>Canvas API를 활용한 의료 영상 진단 마킹 예시</em>
</p>

## 🛠 트러블슈팅 & 기술 구현

### 1. 백엔드 (Back-end) CORS 설정
#### 발생한 문제
- Spring Boot와 FastAPI 서버간 통신 시도 
- 서버 간의 도메인 달라 CORS 문제 발생

#### 해결 방법
- FastAPI에 CORSMiddleware를 추가
- 크로스도메인 통신 허용 설정
- 서버 간 안전적 통신 구현
- 응답 데이터 정상 수신

### 2. 프론트엔드 Canvas API 데이터 소실
#### 발생한 문제
- Canvas 태그에 이미지와 사용자 그림을 동시에 그리도록 구현
- 사용자 화면 변경 시 데이터가 소실되는 과정에서 전체 데이터가 손실

#### 해결 방법
- 이미지와 그림을 각각 별도의 canvas 태그에 분리하여 렌더링
- 화면 변경 전에 데이터를 저장하고 복원하는 작업 추가

### 3. Grad-CAM 모델 구현
#### 발생한 문제
- Resnet 모델로 학습을 진행할 때 사용하던 Grad-CAM은 ViT에 맞게 수정하는 과정에서 문제 발생
- Renet은 2D 활성화 맵이 존재하지만 ViT는 Transformer 기반으로 활성화 맵이 존재하지 않아 차원오류 발생

#### 해결 방법
- reshape_transform 함수를 사용하여 ViT 블록의 출력을 Grad-CAM에서 사용가능한 형태로 변환하여 적용

### 📈 향후 확장 계획
1. 폐 질환 조기/진단을 위한 AI 분석 기능 확장
2. 의료진 업무 효율성 및 의료 서비스 향상
3. X-ray뿐 아니라 MRI, CT까지 확장하여 진단 업무를 위한 의료 서비스 품질 향상
