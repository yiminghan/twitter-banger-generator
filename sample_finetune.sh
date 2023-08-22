if [ -z "$1" ]
then
    echo "Usage: bash sample_finetune.sh [Account name] [Starting Prompt]"
    exit
else
    echo "============Twitter Banger Generator==========="
    echo "Generating Twitter BAnger for account [$1] | prompt: [$2]";
fi

cd nanoGPT
if [ -z "$2" ]
then
echo "====> No input, sample random stuff"
# change the seed in sample.py and see what you get!
python3 sample.py --out_dir=out-$1
else 
echo "====> input detected, completing inference"
python3 sample.py --init_from=resume --start="$2" --num_samples=10 --max_new_tokens=100 --out_dir=out-$1
fi
