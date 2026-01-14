for task in "biomass" "soil_nitrogen" "soil_organic_carbon" "soil_pH" "species"; do
    for architecture in "ConvNeXtV2A" "ScaleMAE" "DINOv3Web" "DINOv3Sat" "SatlasNet" "MPMAE" "TerraMind" "CopernicusFM" "TerraMindS2" "CopernicusFMS2"; do
        for train_percent in 5 50 100; do
            bash run.sh $task $architecture FT $train_percent
            bash run.sh $task $architecture LP $train_percent
        done
    done
    for architecture in "ConvNeXtV2A" "ScaleMAE" "DINOv3Web" "DINOv3Sat" "SatlasNet" "MPMAE" "TerraMind" "CopernicusFM"; do
        bash run.sh $task $architecture JT 100
    done
done
