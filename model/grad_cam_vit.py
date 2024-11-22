import os
import torch
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
import torchvision.models as models
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image


def image_to_base64(image_path: str) -> str:
    """
    이미지 경로에서 이미지를 로드하여 Base64로 변환하는 함수.

    Args:
        image_path (str): 이미지 파일 경로.

    Returns:
        str: Base64로 인코딩된 이미지 문자열.
    """
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_string


def base64_to_image(base64_string: str) -> np.ndarray:
    """
    Base64 문자열을 이미지로 변환하는 함수.

    Args:
        base64_string (str): Base64로 인코딩된 이미지 문자열.

    Returns:
        np.ndarray: OpenCV 형식의 이미지 배열.
    """
    image_bytes = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_bytes))
    return np.array(image)


def reshape_transform(tensor, height=14, width=14):
    """
    ViT(Vision Transformer)의 출력을 Grad-CAM과 호환하는 형식으로 변환하는 함수.
    """
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == "__main__":
    # Grad-CAM 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")

    """ 이미지 경로 설정 """
    image_path = "KakaoTalk_20241115_150513387.png"
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[ERROR] Image not found at path: {image_path}")

    # 1. 이미지 경로에서 Base64로 변환
    base64_image = image_to_base64(image_path)

    # 2. Base64 데이터를 다시 디코딩하여 이미지로 변환
    decoded_image = base64_to_image(base64_image)

    # OpenCV 형식으로 변환
    rgb_img = cv2.cvtColor(decoded_image, cv2.COLOR_RGB2BGR)
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255  # 들여쓰기 수정

    # Vision Transformer(ViT) 모델 로드
    """ 저장된 모델 경로 설정 """
    model_path = './runs/241119_vit_uni_224_64_lr_1e-5/best.pth.tar'  # 저장된 모델 경로

    # Vision Transformer(ViT) 모델 생성
    model = models.vit_b_16(pretrained=False)  # 사전 학습된 모델 가중치 사용하지 않음
    model.heads.head = nn.Linear(model.heads.head.in_features, 14)  # 클래스 수에 맞게 헤드 수정

    # 저장된 체크포인트 로드
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device).eval()  # 모델 평가 모드 설정

    print("[INFO] Pretrained ViT model loaded successfully.")

    # Grad-CAM을 적용할 대상 레이어 설정
    target_layers = [model.encoder.layers[-1].ln_1]  # 마지막 Transformer 블록의 Layer Normalization

    # Grad-CAM 초기화
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

    # Grad-CAM 실행 및 결과 저장
    input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5]).to(device)

    grayscale_cam = cam(input_tensor=input_tensor)
    grayscale_cam = grayscale_cam[0, :]  # 배치에서 첫 번째 이미지 선택

    # 히트맵 정규화
    grayscale_cam[grayscale_cam < 0.2] = 0  # 특정 임계값 이하의 값 제거 (예: 0.2)
    grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min() + 1e-7)  # 최소-최대 정규화

    # 히트맵 생성
    heatmap = np.uint8(255 * grayscale_cam)  # 0~255로 스케일링
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 컬러맵 적용
    heatmap = np.float32(heatmap) / 255.0  # 0~1로 변환

    # 히트맵과 원본 이미지 결합
    alpha = 0.4  # 히트맵 강조 비율
    cam_image = cv2.addWeighted(heatmap, alpha, rgb_img, 1 - alpha, 0)  # 결합
    cam_image = np.clip(cam_image * 255, 0, 255).astype(np.uint8)  # 0~255로 변환

    # 결과 저장
    output_path = "gradcam_output.jpg"
    cv2.imwrite(output_path, cam_image)
    print(f"[INFO] Grad-CAM image saved to: {output_path}")
