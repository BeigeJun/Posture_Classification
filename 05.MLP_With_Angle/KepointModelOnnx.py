import torch
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights

model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1)

dummy_input = torch.randn(1, 3, 480, 640)

torch.onnx.export(
    model,                # 변환할 모델
    dummy_input,          # 모델의 입력 샘플
    "keypointrcnn_cpu.onnx",  # 저장할 ONNX 파일 이름
    export_params=True,   # 모델의 학습된 가중치를 함께 저장
    opset_version=11,     # ONNX opset 버전 (일반적으로 11이 많이 사용됨)
    do_constant_folding=True,  # 상수 폴딩 최적화 수행
    input_names=['input'],   # 입력 텐서 이름
    output_names=['output'], # 출력 텐서 이름
    dynamic_axes={'input': {0: 'batch_size'},    # 동적 배치 크기
                  'output': {0: 'batch_size'}}
)
