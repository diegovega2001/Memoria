from dataclasses import dataclass
from typing import Optional, Tuple
from torchvision import transforms

@dataclass
class TransformConfig:
    grayscale: bool = False
    resize: Optional[Tuple[int, int]] = (224, 224)
    normalize: bool = True
    
    def get_transforms(self):
        transform_list = []
        
        if self.grayscale:
            transform_list.append(transforms.Grayscale(num_output_channels=1))
        
        if self.resize:
            transform_list.append(transforms.Resize(self.resize))
        
        transform_list.append(transforms.ToTensor())
        
        if self.normalize:
            if self.grayscale:
                transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
            else:
                transform_list.append(transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ))
        
        return transforms.Compose(transform_list)
    
    def __call__(self, img):
        transform = self.get_transforms()
        return transform(img)