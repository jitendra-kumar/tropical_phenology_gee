import ee, geemap, os, time
import numpy as np

ee.Initialize()

# set params
COI = 'Brazil'
SOI = 'Amazonas'

# select data bounds
br = (ee.FeatureCollection("FAO/GAUL/2015/level1")
       .filterMetadata('ADM0_NAME', 'equals', COI)
       .filterMetadata('ADM1_NAME', 'equals', SOI)
      )

# set up grid process
def make_grid(region, a_scale):
    lonLat = ee.Image.pixelLonLat()

    lonGrid = (lonLat.select('latitude').multiply(10000000).toInt())
    latGrid = (lonLat.select('longitude').multiply(10000000).toInt())

    grid = (lonGrid.multiply(latGrid).reduceToVectors(
        geometry = region, 
        scale = a_scale, 
        geometryType = 'polygon'))
    
    return(grid)

# make grid
grid_km = make_grid(br, 40000)

# access coordinates of grid squares
grid_dict = grid_km.getInfo()
feats = grid_dict['features']
coord_list = []
for d in feats:
    geom = d['geometry']
    coords = geom['coordinates']
    coord_list.append(coords)
    
# Create a list of several ee.Geometry.Polygons
polys = []
for coord in coord_list:
    poly = ee.Geometry.Polygon(coord)
    polys.append(poly)
    
# Make grid smaller if it's huge
idx = [0]
polys = [ polys[i] for i in idx]

# Make the whole grid a feature collection for export purposes
grid = ee.FeatureCollection(polys)
fc = grid

# set params
SDATE = '2017-01-01'
EDATE = '2020-12-31'

# get daily mean air temp (K) at .25 arc degree res
era5_2mt = (ee.ImageCollection('ECMWF/ERA5/DAILY')
            .select('mean_2m_air_temperature')
            .filter(ee.Filter.date(SDATE, EDATE))
           )

# get total precipitation (m) at .25 arc degree res
era5_tp = (ee.ImageCollection('ECMWF/ERA5/DAILY')
            .select('total_precipitation')
            .filter(ee.Filter.date(SDATE, EDATE))
          )

# set params
STEP = 15
SYEAR = 2017
EYEAR = 2020

# apply 15-day step
years = ee.List.sequence(SYEAR, EYEAR)
step = ee.List.sequence(1, 365, STEP)

clipped_cols = [era5_2mt, era5_tp]
all_cols = []
for a_col in clipped_cols:
    def byYear(y):
        y = ee.Number(y)
        def byStep(d):
            d = ee.Number(d)
            return (a_col
                    .filter(ee.Filter.calendarRange(y, y, 'year'))
                    .filter(ee.Filter.calendarRange(d, d.add(14), 'day_of_year'))
                    .mean()
                    .set('step', [d, y]))
                
        return step.map(byStep)

    col = ee.ImageCollection.fromImages(years.map(byYear).flatten())
    all_cols.append(col)

# make a list of file names
tiles = []
sitename = 'am_0'
for num in range(len(all_cols)):
    index = str(sitename + '_{}'.format(num))
    tiles.append(index)

# export data
polys = [polys[0], polys[0]]
tic1 = time.time()
for a_col, a_tile, poly in zip(all_cols, tiles, polys):
    ilist = a_col.toList(a_col.size())
    for i in range(0,25):
        if len(ee.Image(ilist.get(i)).bandNames().getInfo()) <= 0:
            print("ERROR; No bands found in image index %d... will skip export."%(i))
        else:
            filename = "/mnt/locutus/remotesensing/6ru/era/{}/{}.tif".format(a_tile,i)
            temp_dir = "/mnt/locutus/remotesensing/6ru/era/{}/".format(a_tile)
            if not os.path.exists(temp_dir):
                os.mkdir(temp_dir)
            if os.path.exists(filename):
                print("Image {} already exists. Skipping...".format(i))
                next
            else:
                tic = time.time()
                print("Exporting Image %d"%(i))
                geemap.ee_export_image(ee.Image(ilist.get(i)), 
                                       filename=filename, 
                                       scale=20, 
                                       region=poly, 
                                       file_per_band=False)
                toc = time.time()
                hours, rem = divmod(toc-tic, 3600)
                mins, secs = divmod(rem, 60)
                print("Time elapsed: {:0>2}:{:0>2}:{:05.2f}"
                      .format(int(hours),int(mins),secs))
toc1 = time.time()
hrs1, rem1 = divmod(toc1-tic1, 3600)
mins1, secs1 = divmod(rem1,  60)
print("Total time elapsed: {:0>2}:{:0>2}:{:05.2f}"
      .format(int(hrs1),int(mins1),secs1))