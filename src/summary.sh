

# Command to run: bash summary.sh > summary.txt

python summarize.py --experiment fmnist --regime Balanced
python summarize.py --experiment fmnist-R2 --regime Imbalanced

python summarize.py --experiment svhn --regime Balanced
python summarize.py --experiment svhn-R2 --regime Imbalanced

python summarize.py --experiment cifar10-svhn --regime Balanced
python summarize.py --experiment mnist --regime Balanced
python summarize.py --experiment mnist-fashionmnist --regime Balanced
