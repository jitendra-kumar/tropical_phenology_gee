ncrcat 15d_2017_daily_costarica_panama_1.nc 15d_2018_daily_costarica_panama_1.nc tmp1.nc
ncrcat tmp1.nc 15d_2019_daily_costarica_panama_1.nc tmp2.nc
ncrcat tmp2.nc 15d_2020_daily_costarica_panama_1.nc costarica_panama_15d_2017_2020.nc
rm tmp1.nc tmp2.nc
