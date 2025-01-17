from tabnanny import check

from fastapi import FastAPI, File, UploadFile  # FastAPI 관련 모듈 임포트
from fastapi.responses import JSONResponse
from pydantic import BaseModel  # 데이터 모델링을 위한 Pydantic 임포트
import SimpleITK as sitk  # 의료 이미지 처리를 위한 SimpleITK 임포트
import numpy as np  # 수치 계산을 위한 NumPy 임포트
import cv2  # OpenCV는 이미지 처리 및 컴퓨터 비전을 위한 라이브러리
from PIL import Image  # 이미지 처리를 위한 PIL 라이브러리
import warnings  # 경고 메시지를 제어하는 라이브러리
import logging  # 로깅을 처리하는 라이브러리
import base64  # 바이너리 데이터를 텍스트로 인코딩하는 라이브러리
from io import BytesIO  # 메모리 상에서 바이트 데이터를 처리하기 위한 라이브러리
import tempfile  # 임시 파일 생성을 위한 tempfile 모듈 임포트
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from typing import Dict

import torch
import torchvision
from model import ModelLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

app = FastAPI()
model_loader = ModelLoader("models/best.pth.tar")

# 허용할 출처 설정
origins = [
    "http://localhost:8089",  # 클라이언트가 실행되는 출처 추가
    "https://localhost:8443", # 필요 시 https 출처도 추가
]

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 허용할 출처 목록
    allow_credentials=True,
    allow_methods=["*"],  # 허용할 HTTP 메서드 (GET, POST 등)
    allow_headers=["*"]  # 허용할 HTTP 헤더

)

# 경고 메시지 숨기기 및 로그 레벨 설정 (INFO 이하 메시지는 출력하지 않음)
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

# FastAPI 애플리케이션 인스턴스 생성
# app = FastAPI()

# 응답으로 반환할 데이터 모델 정의 (Pydantic BaseModel 사용)
class DetectionResult(BaseModel):
    message: str       # 처리 결과 메시지
    image: str         # base64로 인코딩된 이미지 데이터
    metadata: Dict     # DICOM 메타데이터 (환자 정보 등)
    model_result : Dict

# DICOM 파일을 처리하여 JSON 형식으로 반환하는 함수
def process_dicom_to_json(dicom_path, image_size=512):
    """
    DICOM 파일을 읽고, 메타데이터와 이미지를 처리하여 JSON 형식으로 변환하는 함수.

    :param dicom_path: 처리할 DICOM 파일의 경로
    :param image_size: 출력 이미지 크기 (기본값: 512x512)
    :return: 메타데이터와 이미지가 포함된 JSON 결과
    """
    try:
        # DICOM 파일 읽기 위한 Reader 생성 및 파일 지정
        reader = sitk.ImageFileReader()
        reader.SetFileName(dicom_path)
        reader.ReadImageInformation()  # DICOM 파일의 메타데이터 읽기

        # DICOM 파일에서 중요한 메타데이터 추출
        metadata = {
            "ptName": reader.GetMetaData("0010|0010").strip() if reader.HasMetaDataKey("0010|0010") else "",  # 환자 이름
            "ptCode": reader.GetMetaData("0010|0020").strip() if reader.HasMetaDataKey("0010|0020") else "",  # 환자 ID
            "ptBirthdate": reader.GetMetaData("0010|0030").strip() if reader.HasMetaDataKey("0010|0030") else "",  # 생년월일
            "ptGen": reader.GetMetaData("0010|0040").strip() if reader.HasMetaDataKey("0010|0040") else ""
        }

        # DICOM 파일의 이미지 데이터를 배열로 변환
        image = sitk.GetArrayFromImage(sitk.ReadImage(dicom_path)).astype('float32').squeeze()

        # 이미지 크기가 지정된 크기와 다르면 크기를 조정
        if image.shape != (image_size, image_size):
            image = cv2.resize(image, (image_size, image_size))

        # 3차원 이미지일 경우 첫 번째 채널만 사용 (흑백 이미지로 가정)
        if len(image.shape) == 3:
            image = image[:, :, 0]

        # 이미지 정규화: 픽셀 값을 0~255 범위로 변환하여 8비트 이미지로 변환
        normalized = (((image - np.min(image)) / (np.max(image) - np.min(image))) * 255).astype(np.uint8)
        img_pil = Image.fromarray(normalized)  # PIL 이미지를 생성

        # PIL 이미지를 Base64로 변환 (텍스트로 인코딩)
        buffered = BytesIO()
        img_pil.save(buffered, format="PNG")  # 이미지 포맷을 PNG로 저장
        img_base64 = base64.b64encode(buffered.getvalue()).decode()  # Base64로 인코딩하여 문자열로 변환

        # 결과 JSON 생성 (메타데이터와 이미지 포함)
        result = {
            "metadata": metadata,  # 메타데이터
            "image_base64": img_base64  # Base64로 인코딩된 이미지 데이터
        }

        return result, img_pil # img_pil 반환

    except Exception as e:
        # 오류 발생 시 로그에 남기고 None 반환
        logging.error(f"Error processing {dicom_path}: {str(e)}")
        return None, None


# albumentations를 이용한 preprocess_image 함수
def preprocess_image(image):
    # albumentations 전처리 파이프라인 구성
    transform = A.Compose([
        A.Resize(224, 224),  # 이미지 크기를 224x224로 조정
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # 정규화
        ToTensorV2()  # 이미지를 텐서로 변환
    ])

    # PIL 이미지를 numpy 배열로 변환 후 전처리 적용
    image_np = np.array(image)  # PIL 이미지를 numpy로 변환
    transformed = transform(image=image_np)  # 전처리 적용
    input_tensor = transformed['image'].unsqueeze(0)  # 배치 차원 추가

    return input_tensor


# DICOM 파일 업로드 및 처리 엔드포인트 정의
@app.post("/dupload",response_model=DetectionResult)
async def process_dicom(file: UploadFile):
    """
    업로드된 DICOM 파일을 처리하여 이미지와 메타데이터를 반환하는 엔드포인트.

    :param file: 업로드된 DICOM 파일
    :return: DetectionResult 모델을 사용하여 처리 결과 반환
    """
    # 업로드된 파일을 임시 파일로 저장하여 처리
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        contents = await file.read()     # 업로드된 파일의 내용을 비동기적으로 읽음
        tmp_file.write(contents)         # 임시 파일에 내용을 씀
        tmp_file_path = tmp_file.name    # 임시 파일의 경로 저장

    # DICOM 파일을 전처리하여 결과 얻기
    result, img_pil = process_dicom_to_json(tmp_file_path)

    if result and img_pil:
        # img_pil을 전처리하여 모델 입력 생성
        input_tensor = preprocess_image(img_pil)  # 전처리 적용

        # 모델 예측
        with torch.no_grad():
            prediction = model_loader.predict(input_tensor)

        # 후처리하여 결과 반환
        # model_result = postprocess_output(prediction)  # 후처리 함수는 사용자가 정의해야 합니다.

        return DetectionResult(
            message="처리가 완료되었습니다.",
            image=result['image_base64'],
            metadata=result['metadata'],
            model_result=model_result
        )
    else:
        return DetectionResult(
            message="DICOM 파일을 처리하는 중 오류가 발생했습니다.",
            image="",
            metadata={},
            model_result={}
        )

# uvicorn으로 이 모듈을 직접 실행할 때 서버를 구동하기 위한 코드
if __name__ == "__main__":
    import uvicorn
    # FastAPI 애플리케이션을 uvicorn으로 실행
    uvicorn.run(app, host="192.168.0.2", port=8000, ssl_certfile="cert.pem", ssl_keyfile="key.pem")
