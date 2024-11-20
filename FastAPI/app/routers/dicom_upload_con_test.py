import logging

from fastapi import UploadFile, APIRouter  # FastAPI 관련 모듈 임포트
from pydantic import BaseModel  # 데이터 모델링을 위한 Pydantic 임포트
import tempfile  # 임시 파일 생성을 위한 tempfile 모듈 임포트
from typing import Dict

import sys
sys.path.append('./MONAI')
import random
from models.resnet import resnet50
from utils_folder.eval_metric import *
import torch
import torch.nn as nn
import torchvision; torchvision.version
import torchvision.models as models

from app.func import process_dicom_to_json, predict_model, gradcam  # process_dicom_to_json과 predict_model 함수 가져오기

router = APIRouter()


# 응답으로 반환할 데이터 모델 정의 (Pydantic BaseModel 사용)
class DetectionResult(BaseModel):
    message: str       # 처리 결과 메시지
    image: str         # base64로 인코딩된 이미지 데이터
    metadata: Dict     # DICOM 메타데이터 (환자 정보 등)
    model_result : Dict # 모델 결과

class GradCAMRequest(BaseModel):
    base64_img: str  # Base64로 인코딩된 입력 이미지

class GradCAMResponse(BaseModel):
    message: str
    image: str


# Args 클래스 정의
filename = '/adamw_weight_updated_imagnet_uni_224_bs128_1e-4'
class Args:
    def __init__(self):
        self.num_class = 14  # 클래스 수
        self.backbone = 'resnet50'  # 백본 모델 선택
        self.log_dir = './runs' + filename  # 로그 디렉토리
        self.img_size = 224  # 이미지 크기
        self.bits = 8  # 비트 수
        self.seed = 50  # 랜덤 시드
        self.w = 4  # 워커 수
        self.resume = True  # 재개 여부
        self.pretrained = './runs' + filename + '/best.pth.tar'  # 프리트레인 모델 경로

args = Args()

# 클래스 리스트 정의
class_list = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
    'Pneumonia', 'Pneumothorax', 'Support Devices'
]

# Device 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 랜덤 시드 설정
if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 모델 선택 및 로드
if args.backbone == 'resnet50':
    model = resnet50(num_classes=args.num_class)

elif args.backbone == 'vit':
    vit = models.vit_b_16(pretrained=True)
    head_in_features = vit.heads.head.in_features
    vit.heads.head = nn.Linear(head_in_features, args.num_class)
    model = vit
else:
    raise ValueError('Invalid backbone: choose resnet50 or vit')

model = model.to(device)

if args.resume:
    checkpoint = torch.load(args.pretrained, map_location=torch.device('cpu'))
    pretrained_dict = checkpoint['state_dict']

    # 입력 채널 크기 확인 및 변환
    if model.conv1.weight.shape[1] == 1:  # Grayscale 입력을 처리하는 경우
        pretrained_conv1 = pretrained_dict['conv1.weight']
        # 평균값을 이용하여 RGB 가중치를 Grayscale 가중치로 변환
        pretrained_dict['conv1.weight'] = pretrained_conv1.mean(dim=1, keepdim=True)

        # 필요한 가중치만 로드
        model_dict = model.state_dict()
        pretrained_dict = {key: value for key, value in pretrained_dict.items() if key in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    '''   
    pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
    model.load_state_dict(pretrained_dict)
    '''
    print("모델 로드 완료")
else:
    raise ValueError('프리트레인 네트워크 경로를 입력해야 합니다.')


# DICOM 파일 업로드 및 처리 엔드포인트 정의
# 모델 예측 라우터
@router.post("/dupload",response_model=DetectionResult, tags=["Medical Analysis"])
async def process_dicom(file: UploadFile):
    """
    업로드된 DICOM 파일을 처리하여 이미지와 메타데이터를 반환하는 엔드포인트.

    Args:
        file (UploadFile): 업로드된 DICOM 파일.

    Returns:
        DetectionResult: 처리 결과.
    """
    try:
        # 업로드된 파일을 임시 파일로 저장하여 처리
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            contents = await file.read()     # 업로드된 파일의 내용을 비동기적으로 읽음
            tmp_file.write(contents)         # 임시 파일에 내용을 씀
            tmp_file_path = tmp_file.name    # 임시 파일의 경로 저장

        # DICOM 파일을 전처리하여 결과 얻기
        # DICOM 처리
        result, img_pil = process_dicom_to_json(tmp_file_path)
        if not result or not img_pil:
            raise ValueError("DICOM 처리 실패")

        # 모델 예측 수행
        try:
            # 모델 예측 수행
            model_result = predict_model(
                img = img_pil,
                model=model,
                device=device,
                num_classes=len(class_list),
                class_list=class_list
            )
        except Exception as e:
            logging.error(f"모델 예측 중 오류 발생: {e}")
            raise ValueError("모델 예측 실패")

        return DetectionResult(
            message="처리가 완료되었습니다.",
            image=result['image_base64'],
            metadata=result['metadata'],
            model_result=model_result
        )
    except Exception as e:
        logging.error(f"파일 처리 중 오류 발생: {e}")
        return {"message": f"Internal Server Error: {e}"}

# Grad-CAM 라우터
@router.post("/resnet_gradcam", response_model=GradCAMResponse, tags=["Medical Analysis"])
def generate_cam(request: GradCAMRequest):
    """
        Grad-CAM 이미지를 생성하고 Base64로 반환합니다.

        Args:
            request (GradCAMRequest): Base64로 인코딩된 입력 이미지를 포함하는 요청.

        Returns:
            GradCAMResponse: 생성된 GradCAM 이미지와 메시지를 포함한 응답.
        """
    try:
        # GradCAM 생성
        logging.info("GradCAM 요청 시작")
        cam_image = gradcam(request.base64_img, model, device)
        logging.info("GradCAM 생성 완료")

        return GradCAMResponse(
            message="Grad-CAM 생성 완료",
            image=cam_image
        )
    except Exception as e:
        return GradCAMResponse(
            message=f"Grad-CAM 생성 중 오류 발생: {str(e)}",
            image=""
        )