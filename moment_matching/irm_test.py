import os
filename = 'summary/test.json'
reg_moments = 500
reg_var = 0.1
lr = 5e-3
w_epoch = 60
os.system(f"python3 MNIST_irm.py {reg_moments} {reg_var} {lr} 1 {w_epoch} 5 {filename}")
