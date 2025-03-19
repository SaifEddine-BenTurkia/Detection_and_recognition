import torch 
print(torch.cuda.is_available())
print("number of gpu" , torch.cuda.device_count())
print("name of gpu" , torch.cuda.get_device_name())