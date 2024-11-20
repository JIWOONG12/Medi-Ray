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

import sys
sys.path.append('./MONAI')
import torch


# 경고 메시지 숨기기 및 로그 레벨 설정 (INFO 이하 메시지는 출력하지 않음)
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)


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
    """
    전처리한 데이터 크기 조절하기
    모델에 넣기위해서는 512*512 인 데이터의 크기를 변경해야한다.
    """
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


def predict_model(img, model, device, num_classes, class_list):
    """
        PIL 이미지와 모델을 입력받아 예측 결과를 반환합니다.

        Args:
            img (PIL.Image): 입력 이미지.
            model (torch.nn.Module): 학습된 모델.
            device (str): 'cuda' 또는 'cpu'.
            num_classes (int): 클래스 수.
            class_list (list): 클래스 이름 리스트.

        Returns:
            dict: 클래스별 예측 결과 (확률값).
        """

    # 이미지 로드 및 전처리
    # RGB 변환 (이미 RGB라면 영향을 미치지 않음)
    img = img.resize((224, 224))  # 모델에 맞게 크기 조정 (예: ResNet은 224x224)
    img = np.array(img) / 255.0  # 정규화

    if img.ndim == 2:  # 흑백 이미지일 경우
        img = np.expand_dims(img, axis=-1)  # 채널 축 추가
    img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float32)  # 텐서로 변환

    model.eval()  # 평가 모드 전환
    with torch.no_grad():
        outputs = torch.sigmoid(model(img_tensor))  # 모델 출력값 -> 확률값 변환

    outputs_list = outputs.cpu().numpy().tolist()
    outputs_dict = {class_list[i]: round(outputs_list[0][i] * 100, 1) for i in range(num_classes)}

    return outputs_dict  # 예측 결과 반환


from monai.visualize import GradCAM

fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)

def gradcam(base64_img, model, device, brightness_factor=1.0):
    logging.info("GradCAM 함수 시작")
    """
        GradCAM 이미지를 생성하고 Base64로 인코딩합니다.

        Args:
            base64_img (str): Base64로 인코딩된 입력 이미지.
            model (torch.nn.Module): 학습된 모델.
            device (str): 'cuda' 또는 'cpu'.
            brightness_factor (float, optional): 밝기 조절 인자. 기본값은 1.0.

        Returns:
            str: Base64로 인코딩된 GradCAM 이미지.
    """
    model.eval() # 평가모드로 전환



    # Base64 이미지를 디코딩하여 PIL 이미지로 변환
    img_data = base64.b64decode(base64_img)
    img = Image.open(BytesIO(img_data)).convert('L')  # 'L' 모드로 변환하여 그레이스케일로 설정
    img = img.resize((224, 224))  # 이미지 크기 조정
    img = np.array(img) / 255.0  # 정규화
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)  # (N, C=1, H, W)
    logging.info("이미지 전처리 완료")

    # GradCAM 객체 생성
    cam = GradCAM(nn_module=model, target_layers='layer4')

    # GradCAM 결과 생성
    result = cam(x=img, class_idx=None, retain_graph=False).squeeze()
    logging.info("GradCAM 생성 완료")
    result = 1 - result  # 히트맵 반전
    heatmap = np.uint8(255 * result)  # 0~255 사이 값으로 변환

    # 컬러맵 적용
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    logging.info("히트맵 생성 및 컬러맵 적용 완료")

    # 임계값 빼기 적용
    heatmap_thresh = heatmap / 255  # 정규화

    # 히트맵과 원본 이미지 결합
    gt_imgs = fn_tonumpy(img)
    original_img = np.repeat(gt_imgs[0][:, :, 0:1], 3, axis=-1)  # 원본 이미지 확장

    # 히트맵의 비중을 높이고 밝기 조절
    superimposed_img = heatmap_thresh * 0.5 + gt_imgs[0] * 0.5  # 히트맵을 더 강조
    superimposed_img = (superimposed_img / brightness_factor) * 255  # 밝기 조절 및 0-255로 변환

    # base64로 전환
    superimposed_img = superimposed_img.astype(np.uint8)

    # 이미지 메모리에 저장
    _, buffer = cv2.imencode('.png', superimposed_img)
    img_bytes = BytesIO(buffer)
    img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
    logging.info("Base64 인코딩 완료")

    return img_base64  # Base64 인코딩된 이미지 반환