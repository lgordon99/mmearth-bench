# initialize gcloud CLI
./google-cloud-sdk/bin/gcloud init

# create conda environment
source ~/.bashrc

conda create --prefix $1/mmearth-bench-env
conda activate $1/mmearth-bench-env
conda install python
conda install pip

pip install earthengine-api
pip install numpy
pip install geojson
pip install pandas
pip install pyproj
pip install geemap
pip install geopandas
pip install rasterio

earthengine authenticate

gcloud auth application-default set-quota-project mmearth-bench
gcloud config set project mmearth-bench
gcloud iam service-accounts create earth-engine-access --display-name earth-engine-access