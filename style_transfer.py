import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------- Image Loading & Preprocessing ---------
def load_image(img_path, max_size=400):
    image = Image.open(img_path).convert('RGB')
    size = max(image.size) if max(image.size) < max_size else max_size
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))
    ])
    image = in_transform(image).unsqueeze(0)
    return image.to(device)

# --------- Display Image ---------
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = image.numpy().transpose(1, 2, 0)
    image = image * (0.229, 0.224, 0.225)
    image = image + (0.485, 0.456, 0.406)
    image = image.clip(0, 1)
    return image

# --------- Load Images ---------
content = load_image("content.jpg")
style = load_image("style.jpg")

# --------- Load Pretrained VGG ---------
vgg = models.vgg19(pretrained=True).features.to(device).eval()
for param in vgg.parameters():
    param.requires_grad_(False)

# --------- Feature Extractor ---------
def get_features(image, model, layers=None):
    if layers is None:
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',  # content layer
            '28': 'conv5_1'
        }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

# --------- Gram Matrix ---------
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    return torch.mm(tensor, tensor.t())

# --------- Extract Features ---------
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# --------- Target Image ---------
target = content.clone().requires_grad_(True).to(device)

# --------- Loss Weights ---------
style_weights = {
    'conv1_1': 1.0,
    'conv2_1': 0.8,
    'conv3_1': 0.5,
    'conv4_1': 0.3,
    'conv5_1': 0.1
}
content_weight = 1e4
style_weight = 1e2

# --------- Optimizer ---------
optimizer = optim.Adam([target], lr=0.003)

# --------- Style Transfer Loop ---------
steps = 2000
for step in range(1, steps + 1):
    target_features = get_features(target, vgg)
    
    # Content loss
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

    # Style loss
    style_loss = 0
    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
        style_loss += layer_style_loss

    # Total loss
    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Progress
    if step % 500 == 0:
        print(f"Step {step}, Total loss: {total_loss.item():.2f}")

# --------- Show Result ---------
final_img = im_convert(target)
plt.imshow(final_img)
plt.axis("off")
plt.title("Stylized Image")
plt.show()

# --------- Save Output ---------
plt.imsave("output.jpg", final_img)
