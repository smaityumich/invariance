import os
filename = 'summary/test.json'
reg_wasserstein = 50
reg_var = 0.01
lr = 5e-4
w_epoch = 10
os.system(f"python3 MNIST_irm.py {reg_wasserstein} {reg_var} {lr} 1 {w_epoch} 5 {filename}")
