import torch
import torch.nn as nn

class SimpleDefectDetectionModel(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleDefectDetectionModel, self).__init__()
        # Simple CNN backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
        # Bounding box head
        self.bbox_regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # (x1, y1, x2, y2)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        class_scores = self.classifier(features)
        bbox_coords = self.bbox_regressor(features)
        return class_scores, bbox_coords

# Create and save a dummy model
if __name__ == "__main__":
    model = SimpleDefectDetectionModel()
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Test the model
    class_scores, bbox_coords = model(dummy_input)
    print(f"Class scores shape: {class_scores.shape}")
    print(f"Bounding box coordinates shape: {bbox_coords.shape}")
    
    # Save the model
    torch.save(model.state_dict(), "simple_defect_model.pth")
    print("Model saved as simple_defect_model.pth") 