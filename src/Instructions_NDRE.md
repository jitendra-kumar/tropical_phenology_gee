## Detection and Attribution of NDRE extremes

**Recommended Conda Env**
Follow the commands one by one:

```
conda create --name da_ndre python=3.7 numpy scipy ipython pandas matplotlib 
conda activate da_ndre
pip install mpi4py
```

---

To execute the code, run the following command on interactive node(s) :

```
time mpiexec -n 30 python detection_disturbance.py \
					-f_ndre ./NDRE/abc.0.ndre \
					-f_pr   ./PR/abc.0.pr \
					-f_tas ./TAS/abc.0.tas
```

    