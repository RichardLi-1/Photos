#!/bin/zsh
# Hyper-parameter sweep for the projection head (each run ~1 min on CPU). Logs to train/logs/.
cd "$(dirname $0)/.."; mkdir -p train/logs
run() { n=$1; shift; .venv/bin/python train/train_head.py --name sw_$n "$@" > train/logs/sw_$n.log 2>&1; echo "$n: $(grep 'best ep' train/logs/sw_$n.log | cut -c1-40) | $(grep 'TEST sw_' train/logs/sw_$n.log | tr -s ' ' | tr '\n' ' ')"; }
case "${1:-all}" in
all)
run base --epochs 20
run t10 --epochs 20 --t 0.1
run t03 --epochs 20 --t 0.03
run lin768 --epochs 20 --dim 768 --depth 1
run mean --epochs 20 --inputs pooled,mean ;&
rest)
run d256 --epochs 20 --dim 256
run wide --epochs 20 --hidden 2048 --drop 0.3
run noise --epochs 20 --noise 0.5 --featdrop 0.1
run ms --epochs 20 --loss multisim --lr 3e-4
run gem --epochs 20 --inputs pooled,gem ;;
esac
case "${1:-all}" in seeds)
for s in 1 2; do
run base_s$s --epochs 20 --seed $s
run ms_s$s --epochs 20 --loss multisim --lr 3e-4 --seed $s
done ;;
esac
case "${1:-all}" in ms2)
run ms_lin --epochs 20 --loss multisim --lr 3e-4 --dim 768 --depth 1
run ms_e40 --epochs 40 --loss multisim --lr 3e-4
run ms_lr1e3 --epochs 20 --loss multisim --lr 1e-3
run ms_mean --epochs 20 --loss multisim --lr 3e-4 --inputs pooled,mean ;;
esac
