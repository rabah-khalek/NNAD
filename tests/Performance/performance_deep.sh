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

for i in `seq 4 10`;
do
    for j in `seq 2 $i`;
    do
        sed -i '' 's/NNarchitecture:.*/NNarchitecture: [1,'$i','$j',1]/' InputCard2.yaml
        for n in `seq 0 50`;
        do
                sed -i '' 's/Seed:.*/Seed: '$((1 + RANDOM % 10000))'/' InputCard2.yaml
                ./main InputCard2.yaml
        done
    done
done

for i in `seq 4 10`;
do
    for j in `seq 2 $i`;
    do
        sed -i '' 's/NNarchitecture:.*/NNarchitecture: [1,'$i','$j',1]/' InputCard3.yaml
        for n in `seq 0 50`;
        do
                sed -i '' 's/Seed:.*/Seed: '$((1 + RANDOM % 10000))'/' InputCard3.yaml
                ./main InputCard3.yaml
        done
    done
done

mv output/*.dat output/Numeric/.
