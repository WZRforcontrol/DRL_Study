import torch

# 打印 PyTorch 版本
print("PyTorch 版本:", torch.__version__)

# 检查 CUDA 是否可用
cuda_available = torch.cuda.is_available()
print("CUDA 是否可用:", cuda_available)

if cuda_available:
    # 打印 CUDA 版本
    print("CUDA 版本:", torch.version.cuda)
    
    # 打印 cuDNN 版本
    print("cuDNN 版本:", torch.backends.cudnn.version())
    
    # 打印 CUDA 库的位置
    print("CUDA 库位置:", torch.cuda.get_device_properties(0))