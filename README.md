# Dynamic GeoCells

This repository implements some ideas from _PIGEON: Predicting Image Geolocations_, 
Haas et al. 2023. It starts from GADM administrative boundary data ([link](https://geodata.ucdavis.edu/gadm/gadm4.1/gadm_410-levels.zip) to 
the needed GeoPackage)
and then uses the distribution of your geotagged training set in order to build geocells (i.e. geometries) that are relevant in terms of point density and structure, increasing the cell resolution where needed.

Notice that the point dataset needs to be a `.csv` with `lat` (latitude) and `lng` (longitude) as columns. In order to get the final result, the order of calls is the following:
1. `init_cells`
2. `merge_cells`
3. `cluster_split`

If needed, the function `geocell_centroid` can be useful to retrieve the centroids for each computed cell.


