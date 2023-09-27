mkdir -p results/exp-FINAL-1/
mkdir -p results/exp-FINAL-2/
mkdir -p results/exp-FINAL-3/
mkdir -p results/exp-FINAL-4/
mkdir -p results/exp-FINAL-5/

python exp-1-observed.py &
python exp-1-onpolicy-baseline.py &

## Baseline
python exp-1-baseline.py --flip_num=0 --flip_seed=0 &
for FLIP_NUM in 25 50 100 200 300 400; do for SEED in 0 42 123 424242 10000; do
python exp-1-baseline.py --flip_num=$FLIP_NUM --flip_seed=$SEED &> /dev/null &
done; done


## Naive implementation

# Naive weighted, Annot π_e
python exp-1-Naive.py --flip_num=0 --flip_seed=0 &
for FLIP_NUM in 25 50 100 200 300 400; do for SEED in 0 42 123 424242 10000; do
python exp-1-Naive.py --flip_num=$FLIP_NUM --flip_seed=$SEED &> /dev/null &
done; done

# Naive UnWeighted, Annot π_e
python exp-1-NaiveUW.py --flip_num=0 --flip_seed=0 &
for FLIP_NUM in 25 50 100 200 300 400; do for SEED in 0 42 123 424242 10000; do
python exp-1-NaiveUW.py --flip_num=$FLIP_NUM --flip_seed=$SEED &> /dev/null &
done; done


## Poposed approach

# Annot π_e
python exp-2-annotEval.py --flip_num=0 --flip_seed=0 &
for FLIP_NUM in 25 50 100 200 300 400; do for SEED in 0 42 123 424242 10000; do
python exp-2-annotEval.py --flip_num=$FLIP_NUM --flip_seed=$SEED &> /dev/null &
done; done

# Annot π_b
python exp-2-annotBeh.py --flip_num=0 --flip_seed=0 &
for FLIP_NUM in 25 50 100 200 300 400; do for SEED in 0 42 123 424242 10000; do
python exp-2-annotBeh.py --flip_num=$FLIP_NUM --flip_seed=$SEED &> /dev/null &
done; done

# Annot zero
python exp-2-annotZero.py --flip_num=0 --flip_seed=0 &
for FLIP_NUM in 25 50 100 200 300 400; do for SEED in 0 42 123 424242 10000; do
python exp-2-annotZero.py --flip_num=$FLIP_NUM --flip_seed=$SEED &> /dev/null &
done; done


##########

## Noisy annotations
for NOISE in '0.0' '0.1' '0.2' '0.3' '0.4' '0.5' '0.6' '0.7' '0.8' '0.9' '1.0'; do
python runs-3-annotEvalNoise.py --flip_num=0 --flip_seed=0 --annot_noise=$NOISE &> /dev/null &
done

for FLIP_NUM in 25 50 100 200 300 400; do for SEED in 0 42 123 424242 10000; do
for NOISE in '0.0' '0.1' '0.2' '0.3' '0.4' '0.5' '0.6' '0.7' '0.8' '0.9' '1.0'; do
python runs-3-annotEvalNoise.py --flip_num=$FLIP_NUM --flip_seed=$SEED --annot_noise=$NOISE &> /dev/null;
done &
done; done



##########

## Missing annotations without and with imputation
for RATIO in '0.0' '0.1' '0.2' '0.3' '0.4' '0.5' '0.6' '0.7' '0.8' '0.9' '1.0'; do
python runs-4-annotEvalMissing.py --flip_num=0 --flip_seed=0 --annot_ratio=$RATIO &> /dev/null;
python runs-4-annotEvalMissingImpute.py --flip_num=0 --flip_seed=0 --annot_ratio=$RATIO &> /dev/null;
done &

for FLIP_NUM in 25 50 100 200 300 400; do for SEED in 0 42 123 424242 10000; do
for RATIO in '0.0' '0.1' '0.2' '0.3' '0.4' '0.5' '0.6' '0.7' '0.8' '0.9' '1.0'; do
python runs-4-annotEvalMissing.py --flip_num=$FLIP_NUM --flip_seed=$SEED --annot_ratio=$RATIO &> /dev/null;
python runs-4-annotEvalMissingImpute.py --flip_num=$FLIP_NUM --flip_seed=$SEED --annot_ratio=$RATIO &> /dev/null;
done &
done; done



##########

## Annot π_b converted approx MDP
mkdir -p './results/runs-2_v4e/'
python runs-5-annotBehConvertedAM.py --flip_num=0 --flip_seed=0 &> /dev/null &
for FLIP_NUM in 25 50 100 200 300 400; do 
for SEED in 0 42 123 424242 10000; do
python runs-5-annotBehConvertedAM.py --flip_num=$FLIP_NUM --flip_seed=$SEED &> /dev/null
done &
done





##########

## Noisy annotations, only 10% annotated without and with imputation
for NOISE in '0.0' '0.1' '0.3' '0.4' '0.5' '0.6' '0.7' '0.8' '0.9' '1.0'; do
python runs-4-annotEvalMissing.py --flip_num=0 --flip_seed=0 --annot_ratio='0.1' --annot_noise=$NOISE &> /dev/null;
python runs-4-annotEvalMissingImpute.py --flip_num=0 --flip_seed=0 --annot_ratio='0.1' --annot_noise=$NOISE &> /dev/null;
done &

for FLIP_NUM in 25 50 100 200 300 400; do for SEED in 0 42 123 424242 10000; do
for NOISE in '0.0' '0.1' '0.3' '0.4' '0.5' '0.6' '0.7' '0.8' '0.9' '1.0'; do
python runs-4-annotEvalMissing.py --flip_num=$FLIP_NUM --flip_seed=$SEED --annot_ratio='0.1' --annot_noise=$NOISE &> /dev/null;
python runs-4-annotEvalMissingImpute.py --flip_num=$FLIP_NUM --flip_seed=$SEED --annot_ratio='0.1' --annot_noise=$NOISE &> /dev/null;
done &
done; done
