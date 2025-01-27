conda install python
conda install pip

pip install beautifulsoup4
pip install cartopy
pip install earthengine-api
pip install h5py
pip install numpy
pip install geemap
pip install geojson
pip install geopandas
pip install pandas
pip install pyproj
pip install rasterio
pip install scipy
pip install wikipedia-api

earthengine authenticate

gcloud auth application-default set-quota-project mmearth-bench
gcloud config set project mmearth-bench
gcloud iam service-accounts create earth-engine-access --display-name earth-engine-access
