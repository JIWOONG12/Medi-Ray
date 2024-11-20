# app/model.py
import torch
import torchvision.models as models

from resnet import resnet50


class ModelLoader:
    def __init__(self, model_path):
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        self.model = resnet50(num_classes=14)  # 또는 사용하는 모델
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()


    def load_model(self, model_path):
        model = models.resnet18(pretrained=False)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # 추론 모드로 설정
        return model

    def predict(self, input_tensor):
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.sigmoid(outputs)  # 시그모이드 적용하여 확률값 반환
            return probabilities[0]  # 배치 차원 제거하고 반환
