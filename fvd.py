import torch.nn as nn
import torch

def load_i3d_model():
    model = torch.hub.load('facebookresearch/pytorchvideo', 'i3d_r50', pretrained=True)
    model.eval()
    return model


def get_i3d_feature_vector(model, video_tensor):
    with torch.no_grad():
        features = model(video_tensor)  # [1, C, T', H', W']
        pooled = nn.AdaptiveAvgPool3d(1)(features)  # [1, C, 1, 1, 1]
    return pooled.squeeze().cpu().numpy()  # [C]

model = load_i3d_model()
test = torch.randn(10, 3, 20, 224, 224)

feat1 = get_i3d_feature_vector(model, test)
print(feat1.shape)