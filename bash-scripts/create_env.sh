# initialize gcloud CLI
./google-cloud-sdk/bin/gcloud init

# create conda environment
source ~/.bashrc

conda create --prefix $1/mmearth-bench-env pip python=3.10.16 -y
