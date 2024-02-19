#!/bin/bash
 
export GISRC=/home/jbk/.grassrc7.chrysaor
grasscr earth_engine costa_rica_panama

UNIVAR_BY_CLUSTER=0
UNIVAR_BY_PROBAV=1

if [ $UNIVAR_BY_CLUSTER -eq 1 ]
then

g.region rast=costa_rica_panama_interp.2017-2022.h10.2017
r.mask rast=costa_rica_panama_interp.2017-2022.h10.2017 maskcats="2 thru 8"
r.univar -et percentile=5,10,90,95 sep=comma map=panama.phen_params.seasonality.2022,costa_rica.phen_params.seasonality.2022 zones=costa_rica_panama_interp.2017-2022.h10.2017 output=costa_rica_panama.phen_params.seasonality.2022.h10.univar 

fi

if [ $UNIVAR_BY_PROBAV -eq 1 ]
then


g.region rast=costa_rica_panama_interp.2017-2022.h10.2017
r.mask rast=costa_rica_panama_interp.2017-2022.h10.2017 maskcats="2 thru 8"
r.univar -et percentile=5,10,90,95 sep=comma map=panama.phen_params.seasonality.2022,costa_rica.phen_params.seasonality.2022 zones=PROBAV_LC100_global_v3.0.1_2019 output=costa_rica_panama.phen_params.seasonality.2022.probav.univar 

sed -i 's/Shrubs/SHR/g' costa_rica_panama.phen_params.seasonality.2022.probav.univar
sed -i 's/Herbaceous_vegetation/HERB/g' costa_rica_panama.phen_params.seasonality.2022.probav.univar
sed -i 's+Cultivated_and_managed_vegetation/agriculture_(cropland)+CROP+g' costa_rica_panama.phen_params.seasonality.2022.probav.univar
sed -i 's+Urban/built_up+URBAN+g' costa_rica_panama.phen_params.seasonality.2022.probav.univar
sed -i 's+Bare/sparse_vegetation+BARREN+g' costa_rica_panama.phen_params.seasonality.2022.probav.univar
sed -i 's/Snow_and_Ice/SNOW/g' costa_rica_panama.phen_params.seasonality.2022.probav.univar
sed -i 's/Permanent_water_bodies/WATER/g' costa_rica_panama.phen_params.seasonality.2022.probav.univar
sed -i 's/Herbaceous_wetland/WET/g' costa_rica_panama.phen_params.seasonality.2022.probav.univar
sed -i 's/Moss_and_Lichen/MOSS/g' costa_rica_panama.phen_params.seasonality.2022.probav.univar
sed -i 's/Closed_forest_evergreen_needle_leaf/CENF/g' costa_rica_panama.phen_params.seasonality.2022.probav.univar
sed -i 's/Closed_forest_evergreen_broad_leaf/CEBF/g' costa_rica_panama.phen_params.seasonality.2022.probav.univar
sed -i 's/Closed_forest_deciduous_needle_leaf/CDNF/g' costa_rica_panama.phen_params.seasonality.2022.probav.univar
sed -i 's/Closed_forest_deciduous_broad_leaf/CDBF/g' costa_rica_panama.phen_params.seasonality.2022.probav.univar
sed -i 's/Closed_forest_mixed/CFM/g' costa_rica_panama.phen_params.seasonality.2022.probav.univar
sed -i 's/Closed_forest_unknown/CFU/g' costa_rica_panama.phen_params.seasonality.2022.probav.univar
sed -i 's/Open_forest_evergreen_needle_leaf/OENF/g' costa_rica_panama.phen_params.seasonality.2022.probav.univar
sed -i 's/Open_forest_evergreen_broad_leaf/OEBF/g' costa_rica_panama.phen_params.seasonality.2022.probav.univar
sed -i 's/Open_forest_deciduous_needle_leaf/ODNF/g' costa_rica_panama.phen_params.seasonality.2022.probav.univar
sed -i 's/Open_forest_deciduous_broad_leaf/ODBF/g' costa_rica_panama.phen_params.seasonality.2022.probav.univar
sed -i 's/Open_forest_mixed/OFM/g' costa_rica_panama.phen_params.seasonality.2022.probav.univar
sed -i 's/Open_forest_unknown/OFU/g' costa_rica_panama.phen_params.seasonality.2022.probav.univar
sed -i 's/Open_sea/SEA/g' costa_rica_panama.phen_params.seasonality.2022.probav.univar
fi
