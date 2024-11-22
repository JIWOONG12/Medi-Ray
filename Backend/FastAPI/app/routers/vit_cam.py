import logging
from pydantic import BaseModel  # 데이터 모델링을 위한 Pydantic 임포트
from fastapi import APIRouter, HTTPException  # FastAPI 관련 모듈 임포트

import sys
sys.path.append('./MONAI')
import cv2
import base64
from io import BytesIO
from utils_folder.eval_metric import *

import torchvision; torchvision.version
import torchvision.models as models
from pytorch_grad_cam import (
    GradCAM,
    GradCAMPlusPlus,
    ScoreCAM,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    EigenGradCAM,
    LayerCAM,
    FullGrad,
)
from pytorch_grad_cam.utils.image import preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit

from app.vit_func import get_args, base64_to_image, reshape_transform

router = APIRouter()

class GradCAMRequest(BaseModel):
    base64_img: str  # Base64로 인코딩된 입력 이미지

class GradCAMResponse(BaseModel):
    message: str
    image: str



# Grad-CAM 라우터
@router.post("/vit_gradcam", response_model=GradCAMResponse, tags=["Grad Cam"])
def vit_gradcam(request: GradCAMRequest):
    try:

        # 요청 데이터를 기반으로 Grad-CAM 설정
        request_data = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # 기본값 설정
            'aug_smooth': False,
            'eigen_smooth': False,
            'method': 'gradcam',
        }

        # Jupyter Notebook 환경을 위한 수정
        # 모델 및 Grad-CAM 설정
        args = get_args(request_data)
        logging.info(f"Parsed arguments: {args.__dict__}")

        # 사용가능한 CAM 메서드 정의
        methods = {"gradcam": GradCAM,
                   "scorecam": ScoreCAM,
                   "gradcam++": GradCAMPlusPlus,
                   "ablationcam": AblationCAM,
                   "xgradcam": XGradCAM,
                   "eigencam": EigenCAM,
                   "eigengradcam": EigenGradCAM,
                   "layercam": LayerCAM,
                   "fullgrad": FullGrad}

        print("모델 로드 완료")

        # Base64 이미지를 디코딩하여 PIL 이미지로 변환
        img = base64_to_image(request.base64_img)
        logging.info("이미지 변환 완료")

        # 사용자가 입력한 메서드가 유효한지 확인
        if args.method not in list(methods.keys()):
            raise Exception(f"method should be one of {list(methods.keys())}")

        # OpenCV 형식으로 변환
        rgb_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        rgb_img = cv2.resize(rgb_img, (224, 224))
        rgb_img = np.float32(rgb_img) / 255

        # Vision Transformer(ViT) 모델 로드
        model_path = './runs/211113_vit_b16_uni_224_32_1e-3/best.pth.tar'  # 저장된 모델 경로

        # Vision Transformer(ViT) 모델 생성
        model = models.vit_b_16(pretrained=False)  # 사전 학습된 모델 가중치 사용하지 않음
        model.heads.head = nn.Linear(model.heads.head.in_features, 14)  # 클래스 수에 맞게 헤드 수정

        # 저장된 체크포인트 로드
        map_location = torch.device(args.device)
        checkpoint = torch.load(model_path, map_location=map_location)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(torch.device(args.device)).eval()  # 모델 평가 모드 설정
        print("[INFO] Pretrained ViT model loaded successfully.")

        # Grad-CAM을 적용할 대상 레이어 설정
        target_layers = [model.encoder.layers[-1].ln_1]  # 마지막 Transformer 블록의 Layer Normalization

        # Grad-CAM 초기화
        if args.method not in methods:
            raise Exception(f"Method {args.method} not implemented")

        if args.method == "ablationcam":
            cam = methods[args.method](model=model,
                                       target_layers=target_layers,
                                       reshape_transform=reshape_transform,
                                       ablation_layer=AblationLayerVit())
        else:
            cam = methods[args.method](model=model,
                                       target_layers=target_layers,
                                       reshape_transform=reshape_transform)

        # Grad-CAM 실행
        input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5]).to(torch.device(args.device))

        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=None,
                            eigen_smooth=args.eigen_smooth,
                            aug_smooth=args.aug_smooth)
        grayscale_cam = grayscale_cam[0, :]  # 배치에서 첫 번째 이미지 선택

        # 히트맵 정규화
        grayscale_cam[grayscale_cam < 0.2] = 0  # 특정 임계값 이하의 값 제거 (예: 0.2)
        grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (
                    grayscale_cam.max() - grayscale_cam.min() + 1e-7)  # 최소-최대 정규화

        # 히트맵 생성
        heatmap = np.uint8(255 * grayscale_cam)  # 0~255로 스케일링
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 컬러맵 적용
        heatmap = np.float32(heatmap) / 255.0  # 0~1로 변환

        # 히트맵과 원본 이미지 결합
        alpha = 0.4  # 히트맵 강조 비율
        cam_image = cv2.addWeighted(heatmap, alpha, rgb_img, 1 - alpha, 0)  # 결합
        cam_image = np.clip(cam_image * 255, 0, 255).astype(np.uint8)  # 0~255로 변환

        # 결과 이미지를 base64로 전환
        # 이미지 메모리에 저장

        _, buffer = cv2.imencode('.png', cam_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')


        return GradCAMResponse(
            message="Grad-CAM 생성 완료",
            image=img_base64
        )

    except Exception as e:

        logging.error(f"Error in vit_gradcam: {e}")

        # FastAPI 예외 처리 개선
        return GradCAMResponse(
            message=f"Grad-CAM 생성 중 오류 발생: {str(e)}",
            image=""
        )