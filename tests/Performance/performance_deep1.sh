for i in `seq 4 10`;
do
    for j in `seq 2 $i`;
    do
        sed -i '' 's/NNarchitecture:.*/NNarchitecture: [1,'$i','$j',1]/' InputCard.yaml
        for n in `seq 0 50`;
        do
                sed -i '' 's/Seed:.*/Seed: '$((1 + RANDOM % 10000))'/' InputCard.yaml
                ./main InputCard.yaml
        done
    done
done