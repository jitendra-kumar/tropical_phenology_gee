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


New Version : `detection_disturbance_with_loc.py`
  * Saves locations of extremes if required , `-save_loc y` or `-save_loc yes`
  * will gerenetate different directory for smooth and interp files

```
time srun -n 30 python /gpfs/alpine/cli137/proj-shared/6ru/proj_analysis/Detection_Attribution/detection_disturbance_with_loc.py \
        -f_ndre /gpfs/alpine/cli137/proj-shared/6ru/proj_analysis/am_0.smooth \
        -f_pr /gpfs/alpine/cli137/proj-shared/6ru/proj_analysis/am_0.precip \
        -f_tas /gpfs/alpine/cli137/proj-shared/6ru/proj_analysis/am_0.temp \
        -attr_typ ano \
        -save_loc yes \
        -save_f_ano no  
```
