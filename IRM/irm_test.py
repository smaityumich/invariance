import os
filename = 'summary/test.json'
reg_wasserstein = 500
reg_var = 0.1
lr = 5e-3
w_epoch = 60
num_steps = 10000
normalize = True
os.system(f"python3 MNIST_irm.py {reg_wasserstein} {reg_var} {lr} 1 {w_epoch} 5 {filename} {num_steps} {normalize}")
