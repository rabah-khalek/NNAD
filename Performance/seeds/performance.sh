for i in `seq 0 50`;
    do
            sed -i '' 's/Seed:.*/Seed: '$i'/' InputCard.yaml
            ./main InputCard.yaml
    done
