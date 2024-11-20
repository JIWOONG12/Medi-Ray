import numpy as np  # 수치 계산을 위한 NumPy 임포트
import albumentations as A
from PIL import Image  # 이미지 처리를 위한 PIL 라이브러리
import SimpleITK as sitk  # 의료 이미지 처리를 위한 SimpleITK 임포트
import cv2  # OpenCV는 이미지 처리 및 컴퓨터 비전을 위한 라이브러리
from io import BytesIO  # 메모리 상에서 바이트 데이터를 처리하기 위한 라이브러리
import base64  # 바이너리 데이터를 텍스트로 인코딩하는 라이브러리
import warnings  # 경고 메시지를 제어하는 라이브러리
import logging  # 로깅을 처리하는 라이브러리
from albumentations.pytorch import ToTensorV2


# 경고 메시지 숨기기 및 로그 레벨 설정 (INFO 이하 메시지는 출력하지 않음)
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)


"""
다이콤에 들어있는
환자의 정보와 다이콤 이미지(png)를 분리하는 함수
"""
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


"""
전처리한 데이터 크기 조절하기
모델에 넣기위해서는 512*512 인 데이터의 크기를 변경해야한다.
"""

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