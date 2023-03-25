#!/bin/bash

# Fitting xgbost is taking long time so for IALE purposes I will sample
# 1000 points in clusters of interest and fit models at those points only

export GISRC=/home/jbk/.grassrc7.chrysaor
grasscr earth_engine costa_rica_panama
#g.region rast=costa_rica_panama_interp.2017-2022.h10.2017

# r.stats -cnpl costa_rica_panama_interp.2017-2022.h10.2017
# 1 1 28520869 8.81%
# 2 2 2758595 0.85%
# 3 3 9109003 2.81%
# 4 4 9093154 2.81%
# 5 5 47317280 14.62%
# 6 6 147409941 45.54%
# 7 7 19147183 5.91%
# 8 8 60305467 18.63%
# 9 9 44671 0.01%
# 10 10 433 0.00%

CREATE_SAMPLE_POINTS=0
SAMPLE_NDRE=0
EXPORT_REPORT=1

clusters=(1 2 3 4 5 6 7 8) # skipping 9 and 10

if [ $CREATE_SAMPLE_POINTS -eq 1 ]
then
for k in ${clusters[@]}
do 
 echo "Working on cluster ${k} in Costa Rica"
 # mask to this cluster only 
 r.mask -r 
 r.mask costa_rica_panama_interp.2017-2022.h10.2017 maskcats=${k}
 # create 10000 random points
 r.random input=costa_rica_panama_interp.2017-2022.h10.2017 cover=costa_rica_panama_interp.2017-2022.h10.2017 npoints=10000 vector=cr_pn_h10_${k} --o

 r.mask -r 
done
fi

if [ $SAMPLE_NDRE -eq 1 ]
then
g.region rast=costa_rica_panama_interp.2017-2022.h10.2017

 for((y=2017; y<=2022; y++)) do
  for((p=0; p<25; p++)) do
   echo "Year ${y} Period ${p}"
   r.patch in=costa_rica_interp_${y}_`printf %02d "$p"`,panama_interp_${y}_`printf %02d "$p"` out=ndre_tmp

   for k in ${clusters[@]}
   do 
    v.db.addcolumn cr_pn_h10_${k} column="n${y}_${p} double precision"
    v.what.rast map=cr_pn_h10_${k} raster=ndre_tmp column=n${y}_${p}
   done

   g.remove -f type=rast  name=ndre_tmp
  done
 done
fi

if [ $EXPORT_REPORT -eq 1 ]
then
 for k in ${clusters[@]}
 do 
  v.report cr_pn_h10_${k} option=coor > cr_pn_h10_${k}.report
 done
fi
#r.random input=costa_rica_panama_interp.2017-2022.h10.2017 cover=costa_rica_panama_interp.2017-2022.h10.2017 npoints=1000 vector=cr_pn_h10_test --o
#
#for((y=2017; y<=2017; y++)) do
# for((p=0; p<25; p++)) do
#  v.db.addcolumn cr_pn_h10_test column="n${y}_${p} double precision"
#  r.patch in=costa_rica_interp_${y}_`printf %02d "$p"`,panama_interp_${y}_`printf %02d "$p"` out=ndre_tmp 
#  v.what.rast map=cr_pn_h10_test raster=ndre_tmp column=n${y}_${p}
#  g.remove -f type=rast  name=ndre_tmp
# done
#done
