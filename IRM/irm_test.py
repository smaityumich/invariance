import os
filename = 'summary/test.json'
reg_wasserstein = 0e-2
reg_var = 0.1
lr = 1e-3
w_epoch = 1000
num_steps = 10000
normalize = True
os.system(f"python3 MNIST_irm.py {reg_wasserstein} {reg_var} {lr} 1 {w_epoch} 5 {filename} {num_steps} {normalize}")
