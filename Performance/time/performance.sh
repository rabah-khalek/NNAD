for i in `seq 5 50`;
do
    sed -i '' 's/NNarchitecture:.*/NNarchitecture: [1,'$i',1]/' InputCard.yaml
    for i in `seq 0 10`;
    do
            sed -i '' 's/Seed:.*/Seed: '$i'/' InputCard.yaml
            ./main InputCard.yaml
    done
done