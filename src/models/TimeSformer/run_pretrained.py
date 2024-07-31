import torch
from timesformer.models.vit import TimeSformer

model = TimeSformer(img_size=448, num_classes=400, num_frames=64, attention_type='divided_space_time',  pretrained_model='pretrained_model/TimeSformer_divST_64x32_224_HowTo100M.pyth')

dummy_video = torch.randn(2, 3, 64, 448, 448) # (batch x channels x frames x height x width)

pred = model(dummy_video,) # (2, 400)

print(pred)