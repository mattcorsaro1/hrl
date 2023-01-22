# Oscar commands
`ssh <username>@ssh.ccv.brown.edu`

## Checking jobs
`myq`, `allq`, and `allq 3090-gcondo` to check job status. `scancel` to cancel. `myjobinfo` to see info.

## File storage
`~/scratch` is deleted after 30 days, `~/data/mcorsaro` is for long-term storage. Check with `myquota`.

# Setup hrl on the cluster
(This will be tested and moved to README.md)
Start with rainbow setup/virtualenv (see TaskOrientedGrasping/notes/ccv_setup.md)

(First, use old venv)
git clone git@github.com:mattcorsaro1/hrl.git
cd hrl
git checkout matt/init-robot
submodule update --init --recursive
cd rainbow_RBFDQN
pip install .
pip freeze | grep rainbow
Install d4rl https://github.com/Farama-Foundation/D4RL
(Installing collected packages: pybullet, mjrl, termcolor, h5py, mujoco, click, dm_control, D4RL)
pip install seeding
pip install scikit-learn
pip install ipdb
pip install treelib

# Running hrl on the cluster (new)
## Debug
### old_main.py
```
interact -g 1 -q 3090-gcondo -m 40G -n 4 -t 2:00:00
source ~/installations/virtualenvs/mujoco/bin/activate
cd ~/Software/hrl
python -m hrl --experiment_name hrl_test --results_dir ~/scratch/results --device 'cuda:0' --environment antmaze-umaze-v0 --seed 0 --use_value_function --use_global_value_function --episodes 1000 --use_diverse_starts --use_global_option_subgoals --init_classifier_type dist-clf  --lr_c 3e-4 --lr_a 3e-4 --logging_frequency 5 --agent rbf --gestation_period 2
```

```
interact -g 1 -q 3090-gcondo -m 40G -n 4 -t 2:00:00
cd ~/Software/hrl
source ~/installations/virtualenvs/mujoco/bin/activate
python -m hrl --experiment_name hrl_test_0 --results_dir ~/scratch/results --device 'cuda:0' --environment antmaze-umaze-v0 --seed 0 --episodes 1000 --use_dense_rewards --use_HER
```

Example of 4 things to run:
```
onager prelaunch --exclude-job-id +jobname ant_dense_her +command "python -m hrl --experiment_name ant_dense_her --results_dir ~/scratch/results --device 'cuda:0' --environment antmaze-umaze-v0 --episodes 3000 --use_dense_rewards --use_HER" +arg --seed 0 1 +no-tag-arg --seed

onager prelaunch --exclude-job-id +jobname ant_sparse_her +command "python -m hrl --experiment_name ant_sparse_her --results_dir ~/scratch/results --device 'cuda:0' --environment antmaze-umaze-v0 --episodes 3000 --use_HER" +arg --seed 0 1 +no-tag-arg --seed

onager prelaunch --exclude-job-id +jobname ant_dense +command "python -m hrl --experiment_name ant_dense --results_dir ~/scratch/results --device 'cuda:0' --environment antmaze-umaze-v0 --episodes 3000 --use_dense_rewards" +arg --seed 0 1 +no-tag-arg --seed

onager prelaunch --exclude-job-id +jobname ant_sparse +command "python -m hrl --experiment_name ant_sparse --results_dir ~/scratch/results --device 'cuda:0' --environment antmaze-umaze-v0 --episodes 3000" +arg --seed 0 1 +no-tag-arg --seed

onager launch --backend slurm --jobname ant_dense_her --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
onager launch --backend slurm --jobname ant_sparse_her --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
onager launch --backend slurm --jobname ant_dense --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
onager launch --backend slurm --jobname ant_sparse --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
```

## Onager
cd ~/Software/hrl
source ~/installations/virtualenvs/mujoco/bin/activate

# Running rainbow on the cluster (old)
cd Software/rainbow_RBFDQN/
pip install .
onager prelaunch --exclude-job-id +jobname switch_clf_uw_20k +command "python -u experiments/experiment.py --hyper_parameter_name Switch --experiment_name ~/scratch/results/switch_clf_uw_20k --run_title rainbow --double True --distributional True --per True --log --task switch --reward_sparse True --gravity True --lock_gripper True --sample_method classifier_unweighted" +arg --seed 0 +no-tag-arg --seed
onager launch --backend slurm --jobname switch_clf_uw_20k --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
python scripts/plot_results_matt.py
source ~/installations/virtualenvs/mujoco/bin/activate
