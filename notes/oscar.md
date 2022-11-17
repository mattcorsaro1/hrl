# Oscar commands
`ssh <username>@ssh.ccv.brown.edu`

## Checking jobs
`myq`, `allq`, and `allq 3090-gcondo` to check job status. `scancel` to cancel.

## File storage
`~/scratch` is deleted after 30 days, `~/data/mcorsaro` is for long-term storage. Check with `myquota`.

# Running hrl on the cluster (new)

# Running rainbow on the cluster (old)
cd Software/rainbow_RBFDQN/
pip install .
onager prelaunch --exclude-job-id +jobname switch_clf_uw_20k +command "python -u experiments/experiment.py --hyper_parameter_name Switch --experiment_name ~/scratch/results/switch_clf_uw_20k --run_title rainbow --double True --distributional True --per True --log --task switch --reward_sparse True --gravity True --lock_gripper True --sample_method classifier_unweighted" +arg --seed 0 +no-tag-arg --seed
onager launch --backend slurm --jobname switch_clf_uw_20k --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
python scripts/plot_results_matt.py
source ~/installations/virtualenvs/mujoco/bin/activate
