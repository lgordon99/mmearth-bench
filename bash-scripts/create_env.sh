# initialize gcloud CLI
./google-cloud-sdk/bin/gcloud init

# create conda environment
source ~/.bashrc

conda create --prefix $1/mmearth-bench-env
