import base64

import numpy as np
from PIL import Image
from io import BytesIO

import sys
sys.path.append('./MONAI')
import torch



def get_args(request_data=None):
    """
     HTTP 요청 데이터를 처리하여 Grad-CAM 설정 값을 반환합니다.
    """
    # Jupyter Notebook 환경 확인
    class Args:
        def __init__(self):
            # 요청 데이터가 제공된 경우 이를 사용, 기본값은 설정
            self.device = request_data.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            self.aug_smooth = request_data.get('aug_smooth', False)
            self.eigen_smooth = request_data.get('eigen_smooth', False)
            self.method = request_data.get('method', 'gradcam')

    return Args()



def reshape_transform(tensor, height=14, width=14):
    """
    ViT(Vision Transformer)의 출력을 Grad-CAM과 호환하는 형식으로 변환하는 함수.
    - Transforemr의 토큰을 2D height x width 크기로 변환
    - CNN 형식과 유사하게 채널이 첫 번째 축으로 오게 전치(transpose)
    """
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # CNN과 유사한 순서로 변환 : 채널(C) -> 높이(H) -> 너비(W)
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result



def base64_to_image(base64_img) -> np.ndarray:
    """
    Base64 문자열을 이미지로 변환하는 함수.

    Args:
        base64_string (str): Base64로 인코딩된 이미지 문자열.

    Returns:
        np.ndarray: OpenCV 형식의 이미지 배열.
    """
    img_data = base64.b64decode(base64_img)
    image = Image.open(BytesIO(img_data)).convert("RGB")
    image_np = np.array(image)

    return image_np
