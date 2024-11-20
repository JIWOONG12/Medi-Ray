from fastapi import UploadFile, APIRouter  # FastAPI 관련 모듈 임포트
from pydantic import BaseModel  # 데이터 모델링을 위한 Pydantic 임포트
import tempfile  # 임시 파일 생성을 위한 tempfile 모듈 임포트
from typing import Dict
import torch
from model import ModelLoader

from app import func

router = APIRouter()
model_loader = ModelLoader("models/best.pth.tar")


# 응답으로 반환할 데이터 모델 정의 (Pydantic BaseModel 사용)
class DetectionResult(BaseModel):
    message: str       # 처리 결과 메시지
    image: str         # base64로 인코딩된 이미지 데이터
    metadata: Dict     # DICOM 메타데이터 (환자 정보 등)
    model_result : Dict


# DICOM 파일 업로드 및 처리 엔드포인트 정의
@router.post("/dupload",response_model=DetectionResult)
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
    result, img_pil = pre_func.process_dicom_to_json(tmp_file_path)

    if result and img_pil:
        # img_pil을 전처리하여 모델 입력 생성
        input_tensor = pre_func.preprocess_image(img_pil)  # 전처리 적용

        # 모델 예측
        with torch.no_grad():
            prediction = model_loader.predict(input_tensor)

        # 후처리하여 결과 반환
        #model_result = postprocess_output(prediction)  # 후처리 함수는 사용자가 정의해야 합니다.

        return DetectionResult(
            message="처리가 완료되었습니다.",
            image=result['image_base64'],
            metadata=result['metadata'],
            #model_result=model_result
        )
    else:
        return DetectionResult(
            message="DICOM 파일을 처리하는 중 오류가 발생했습니다.",
            image="",
            metadata={},
            model_result={}
        )
