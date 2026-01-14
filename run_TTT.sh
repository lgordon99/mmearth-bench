for task in "biomass" "soil_nitrogen" "soil_organic_carbon" "soil_pH" "species"; do
    for architecture in "ConvNeXtV2A" "ScaleMAE" "DINOv3Web" "DINOv3Sat" "SatlasNet" "MPMAE" "TerraMind" "CopernicusFM"; do
        bash run.sh $task $architecture JT-TTT 100
        bash run.sh $task $architecture JT-TTT-Geo 100
    done
done
