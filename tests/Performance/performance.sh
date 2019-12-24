#for i in `seq 56 100`;
#do
#    sed -i '' 's/NNarchitecture:.*/NNarchitecture: [1,'$i',1]/' InputCard.yaml
#    for j in `seq 0 100`;
#    do
#            sed -i '' 's/Seed:.*/Seed: '$((1 + RANDOM % 10000))'/' InputCard.yaml
#            ./main InputCard.yaml
#    done
#done

#mv output/*.dat output/Analytic/.

for i in `seq 5 83`;
do
    sed -i '' 's/NNarchitecture:.*/NNarchitecture: [1,'$i',1]/' InputCard2.yaml
    for j in `seq 0 100`;
    do
            sed -i '' 's/Seed:.*/Seed: '$((1 + RANDOM % 10000))'/' InputCard2.yaml
            ./main InputCard2.yaml
    done
done

mv output/*.dat output/Automatic/.

for i in `seq 5 83`;
do
    sed -i '' 's/NNarchitecture:.*/NNarchitecture: [1,'$i',1]/' InputCard3.yaml
    for j in `seq 0 100`;
    do
            sed -i '' 's/Seed:.*/Seed: '$((1 + RANDOM % 10000))'/' InputCard3.yaml
            ./main InputCard3.yaml
    done
done

mv output/*.dat output/Numeric/.

#for i in `seq 10 20`;
#do
#    for k in `seq 10 20`;
#    do
#        sed -i '' 's/NNarchitecture:.*/NNarchitecture: [1,'$i','$k',1]/' InputCard.yaml
#        for j in `seq 0 100`;
#        do
#                sed -i '' 's/Seed:.*/Seed: '$((1 + RANDOM % 10000))'/' InputCard.yaml
#                ./main InputCard.yaml
#        done
#    done
#done