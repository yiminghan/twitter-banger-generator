if [ -z "$1" ]; then
    echo "Usage: bash fintune_gpt2.sh [Account name]"
    exit
else
    echo "============Twitter Banger Generator==========="
    echo "Finetune GPT2 for account [$1]";
fi

mkdir -p "nanoGPT/data/$1"
# probably Unnesscary copy but ¯\_(ツ)_/¯
cp "nanoGPT/data/shakespeare/prepare.py" "nanoGPT/data/$1/prepare.py"
cp "scrap/$1.txt" "nanoGPT/data/$1/input.txt"
cp "finetune.py" "nanoGPT/config/finetune.py"

python3 nanoGPT/data/$1/prepare.py

cd nanoGPT

# My broke ass computer can only run gpt2-medium, change this to 'gpt2-large', 'gpt2-xl' if you have a more powerful GPU
python3 train.py config/finetune.py --out_dir=out-$1 --dataset=$1  --init_from=gpt2-medium

   



