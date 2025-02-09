
import os
import numpy as np
import itertools
 

job_file = 'submit.sbat'

# Experiment 1
reg_w = np.array(range(1, 21))
reg_v = [0.01]
#reg_v = np.array(range(1, 11))/50
lrs = np.array([1e-4])
iters = range(1)
sh_itr = [5, 10, 20, 40]

os.system('touch summary/out_mnist8.json')



for reg_wasserstein, reg_var, lr, _, sinkhorn_iter in itertools.product(reg_w, reg_v, lrs, iters, sh_itr):
    os.system(f'touch {job_file}')

        
    with open(job_file,'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines(f"#SBATCH --job-name=invar.job\n")
        fh.writelines('#SBATCH --nodes=1\n')
        fh.writelines('#SBATCH --cpus-per-task=1\n')
        fh.writelines('#SBATCH --mem-per-cpu=2gb\n')
        fh.writelines("#SBATCH --time=03:00:00\n")
        fh.writelines("#SBATCH --account=yuekai1\n")
        fh.writelines("#SBATCH --mail-type=NONE\n")
        fh.writelines("#SBATCH --mail-user=smaity@umich.edu\n")
        fh.writelines('#SBATCH --partition=standard\n')
        fh.writelines(f"python3 MNIST_expt.py {reg_wasserstein} {reg_var} {lr} 1 1 {sinkhorn_iter}")


    os.system("sbatch %s" %job_file)
    os.system(f'rm {job_file}')


