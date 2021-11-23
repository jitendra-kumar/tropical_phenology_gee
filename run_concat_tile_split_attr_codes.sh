#!/bin/bash

filepath_head="/mnt/locutus/remotesensing/tropics/anomalies_detection/"
region="puerto_rico"
out_path_main=${filepath_head}${region}"_attr_summary"
out_path_tile_split=${out_path_main}"/tile_split_files"
mkdir $out_path_main
mkdir $out_path_tile_split

for i in {000..013}
do
 for j in {00..15}
 do
  echo "Concatenating tile $i - split $j"
  sub_dir_tile=${filepath_head}${region}"/"${region}"_"${i}"/"
  sub_dir_tile_split=${sub_dir_tile}${region}"_"${i}"_"${j}"/"

  cat ${sub_dir_tile_split}attr_ano_lag_00*.csv > ${out_path_tile_split}"/"${i}"_"${j}"_attr_ano_lag_00.csv"
  cat ${sub_dir_tile_split}attr_ano_lag_01*.csv > ${out_path_tile_split}"/"${i}"_"${j}"_attr_ano_lag_01.csv"
  cat ${sub_dir_tile_split}attr_ano_lag_02*.csv > ${out_path_tile_split}"/"${i}"_"${j}"_attr_ano_lag_02.csv"
  cat ${sub_dir_tile_split}attr_ano_lag_03*.csv > ${out_path_tile_split}"/"${i}"_"${j}"_attr_ano_lag_03.csv"
  cat ${sub_dir_tile_split}attr_ano_lag_04*.csv > ${out_path_tile_split}"/"${i}"_"${j}"_attr_ano_lag_04.csv"
  cat ${sub_dir_tile_split}attr_ano_lag_05*.csv > ${out_path_tile_split}"/"${i}"_"${j}"_attr_ano_lag_05.csv"

 done
done

#Concatenating all tile split files to one file per lag
cat ${out_path_tile_split}"/"*_attr_ano_lag_00.csv > ${out_path_main}"/All_"${region}"_attr_ano_lag_00.csv"
cat ${out_path_tile_split}"/"*_attr_ano_lag_01.csv > ${out_path_main}"/All_"${region}"_attr_ano_lag_01.csv"
cat ${out_path_tile_split}"/"*_attr_ano_lag_02.csv > ${out_path_main}"/All_"${region}"_attr_ano_lag_02.csv"
cat ${out_path_tile_split}"/"*_attr_ano_lag_03.csv > ${out_path_main}"/All_"${region}"_attr_ano_lag_03.csv"
cat ${out_path_tile_split}"/"*_attr_ano_lag_04.csv > ${out_path_main}"/All_"${region}"_attr_ano_lag_04.csv"
cat ${out_path_tile_split}"/"*_attr_ano_lag_05.csv > ${out_path_main}"/All_"${region}"_attr_ano_lag_05.csv"
