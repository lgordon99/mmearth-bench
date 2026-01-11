conda install -c conda-forge gdal=3.10 proj geos geotiff -y
pip install -r requirements.txt
pip install --no-deps "git+https://github.com/IBM/terratorch.git"

earthengine authenticate

gcloud auth application-default set-quota-project mmearth-bench
gcloud config set project mmearth-bench
gcloud iam service-accounts create earth-engine-access --display-name earth-engine-access
