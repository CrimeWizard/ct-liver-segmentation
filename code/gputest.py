import torch, time

size = 8000  # larger matrix favors GPU
loops = 20

x_gpu = torch.randn(size, size, device="cuda")
torch.cuda.synchronize()
t0 = time.time()
for _ in range(loops):
    y = torch.matmul(x_gpu, x_gpu)
torch.cuda.synchronize()
t1 = time.time()
print(f"GPU {loops}x took {t1 - t0:.3f}s")

x_cpu = torch.randn(size, size)
t0 = time.time()
for _ in range(loops):
    y_cpu = torch.matmul(x_cpu, x_cpu)
t1 = time.time()
print(f"CPU {loops}x took {t1 - t0:.3f}s")
