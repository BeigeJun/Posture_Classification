# import gputil
import torch

# gpu = gputil.getGPUs()[0]
# print(gpu.temperature)

memory_used = torch.cuda.memory_reserved() / (1024**3)
print(memory_used)