import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU count: {torch.cuda.device_count()}")


# # насtройка device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # Модель на GPU
# model = AudioClassifier().to(device)

# # Данные на GPU (в training loop)
# for specs, labels in dataloader:
#     specs = specs.to(device)
#     labels = labels.to(device)
    
#     outputs = model(specs)
#     # ...