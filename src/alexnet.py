from torchvision.models import alexnet, AlexNet_Weights

# Step 1: Initialize model with the best available weights
weights = AlexNet_Weights.DEFAULT
model = alexnet(weights=weights)

print(model)

