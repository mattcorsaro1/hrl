```
onager prelaunch --exclude-job-id +jobname switch_clf_her_trial_0 +command "python -m hrl --experiment_name switch_clf_her_trial_0 --results_dir ~/scratch/results --device 'cuda:0' --environment switch --episodes 20000 --sample_method classifier --use_HER" +arg --seed 0 1 2 3 4 +no-tag-arg --seed
onager prelaunch --exclude-job-id +jobname switch_uw_clf_her_trial_0 +command "python -m hrl --experiment_name switch_uw_clf_her_trial_0 --results_dir ~/scratch/results --device 'cuda:0' --environment switch --episodes 20000 --sample_method classifier_unweighted --use_HER" +arg --seed 0 1 2 3 4 +no-tag-arg --seed
onager prelaunch --exclude-job-id +jobname switch_oracle_her_trial_0 +command "python -m hrl --experiment_name switch_oracle_her_trial_0 --results_dir ~/scratch/results --device 'cuda:0' --environment switch --episodes 20000 --sample_method oracle --use_HER" +arg --seed 0 1 2 3 4 +no-tag-arg --seed
onager prelaunch --exclude-job-id +jobname switch_baseline_her_trial_0 +command "python -m hrl --experiment_name switch_baseline_her_trial_0 --results_dir ~/scratch/results --device 'cuda:0' --environment switch --episodes 20000 --sample_method random --use_HER" +arg --seed 0 1 2 3 4 +no-tag-arg --seed

onager prelaunch --exclude-job-id +jobname door_clf_trial_0 +command "python -m hrl --experiment_name door_clf_trial_0 --results_dir ~/scratch/results --device 'cuda:0' --environment door --episodes 20000 --sample_method classifier" +arg --seed 0 1 2 3 4 +no-tag-arg --seed
onager prelaunch --exclude-job-id +jobname door_uw_clf_trial_0 +command "python -m hrl --experiment_name door_uw_clf_trial_0 --results_dir ~/scratch/results --device 'cuda:0' --environment door --episodes 20000 --sample_method classifier_unweighted" +arg --seed 0 1 2 3 4 +no-tag-arg --seed
onager prelaunch --exclude-job-id +jobname door_oracle_trial_0 +command "python -m hrl --experiment_name door_oracle_trial_0 --results_dir ~/scratch/results --device 'cuda:0' --environment door --episodes 20000 --sample_method oracle" +arg --seed 0 1 2 3 4 +no-tag-arg --seed
onager prelaunch --exclude-job-id +jobname door_baseline_trial_0 +command "python -m hrl --experiment_name door_baseline_trial_0 --results_dir ~/scratch/results --device 'cuda:0' --environment door --episodes 20000 --sample_method random" +arg --seed 0 1 2 3 4 +no-tag-arg --seed

onager launch --backend slurm --jobname switch_clf_her_trial_0 --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
onager launch --backend slurm --jobname switch_uw_clf_her_trial_0 --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
onager launch --backend slurm --jobname switch_oracle_her_trial_0 --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
onager launch --backend slurm --jobname switch_baseline_her_trial_0 --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
onager launch --backend slurm --jobname door_clf_trial_0 --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
onager launch --backend slurm --jobname door_uw_clf_trial_0 --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
onager launch --backend slurm --jobname door_oracle_trial_0 --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
onager launch --backend slurm --jobname door_baseline_trial_0 --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
```

```
onager prelaunch --exclude-job-id +jobname switch_clf_uw_her_3 +command "python -m hrl --experiment_name switch_clf_uw_her_3 --results_dir ~/scratch/results --device 'cuda:0' --environment switch --episodes 20000 --sample_method classifier_unweighted --use_HER" +arg --seed 0 +no-tag-arg --seed
onager prelaunch --exclude-job-id +jobname door_clf_3 +command "python -m hrl --experiment_name door_clf_3 --results_dir ~/scratch/results --device 'cuda:0' --environment door --episodes 20000 --sample_method classifier" +arg --seed 0 +no-tag-arg --seed
onager prelaunch --exclude-job-id +jobname door_clf_her_3 +command "python -m hrl --experiment_name door_clf_her_3 --results_dir ~/scratch/results --device 'cuda:0' --environment door --episodes 20000 --sample_method classifier --use_HER" +arg --seed 0 +no-tag-arg --seed
onager prelaunch --exclude-job-id +jobname door_clf_uw_3 +command "python -m hrl --experiment_name door_clf_uw_3 --results_dir ~/scratch/results --device 'cuda:0' --environment door --episodes 20000 --sample_method classifier_unweighted" +arg --seed 0 +no-tag-arg --seed
onager prelaunch --exclude-job-id +jobname door_clf_uw_her_3 +command "python -m hrl --experiment_name door_clf_uw_her_3 --results_dir ~/scratch/results --device 'cuda:0' --environment door --episodes 20000 --sample_method classifier_unweighted --use_HER" +arg --seed 0 +no-tag-arg --seed

onager launch --backend slurm --jobname switch_clf_her_3 --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
onager launch --backend slurm --jobname switch_clf_uw_her_3 --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
onager launch --backend slurm --jobname door_clf_3 --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
onager launch --backend slurm --jobname door_clf_her_3 --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
onager launch --backend slurm --jobname door_clf_uw_3 --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
onager launch --backend slurm --jobname door_clf_uw_her_3 --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo

onager prelaunch --exclude-job-id +jobname door_oracle_3 +command "python -m hrl --experiment_name door_oracle_3 --results_dir ~/scratch/results --device 'cuda:0' --environment door --episodes 20000 --sample_method oracle" +arg --seed 0 +no-tag-arg --seed
onager launch --backend slurm --jobname door_oracle_3 --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
```

door no her
switch her

# Grid Search
- Door
- HER + not
- Seed 0 + 1
- lr_c 3e-3, 3e-4, 3e-5
- lr_a 3e-3, 3e-4, 3e-5
- sample_method random
```
onager prelaunch --exclude-job-id +jobname door_gs1_0_HER_3e-3_3e-3 +command "python -m hrl --experiment_name door_gs1_0_HER_3e-3_3e-3 --results_dir ~/scratch/results --device 'cuda:0' --environment door --episodes 20000 --sample_method random --use_HER --lr_c 3e-3 --lr_a 3e-3" +arg --seed 0 1 +no-tag-arg --seed
onager prelaunch --exclude-job-id +jobname door_gs1_1_HER_3e-3_3e-4 +command "python -m hrl --experiment_name door_gs1_1_HER_3e-3_3e-4 --results_dir ~/scratch/results --device 'cuda:0' --environment door --episodes 20000 --sample_method random --use_HER --lr_c 3e-3 --lr_a 3e-4" +arg --seed 0 1 +no-tag-arg --seed
onager prelaunch --exclude-job-id +jobname door_gs1_2_HER_3e-3_3e-5 +command "python -m hrl --experiment_name door_gs1_2_HER_3e-3_3e-5 --results_dir ~/scratch/results --device 'cuda:0' --environment door --episodes 20000 --sample_method random --use_HER --lr_c 3e-3 --lr_a 3e-5" +arg --seed 0 1 +no-tag-arg --seed
onager prelaunch --exclude-job-id +jobname door_gs1_3_HER_3e-4_3e-3 +command "python -m hrl --experiment_name door_gs1_3_HER_3e-4_3e-3 --results_dir ~/scratch/results --device 'cuda:0' --environment door --episodes 20000 --sample_method random --use_HER --lr_c 3e-4 --lr_a 3e-3" +arg --seed 0 1 +no-tag-arg --seed
onager prelaunch --exclude-job-id +jobname door_gs1_4_HER_3e-4_3e-4 +command "python -m hrl --experiment_name door_gs1_4_HER_3e-4_3e-4 --results_dir ~/scratch/results --device 'cuda:0' --environment door --episodes 20000 --sample_method random --use_HER --lr_c 3e-4 --lr_a 3e-4" +arg --seed 0 1 +no-tag-arg --seed
onager prelaunch --exclude-job-id +jobname door_gs1_5_HER_3e-4_3e-5 +command "python -m hrl --experiment_name door_gs1_5_HER_3e-4_3e-5 --results_dir ~/scratch/results --device 'cuda:0' --environment door --episodes 20000 --sample_method random --use_HER --lr_c 3e-4 --lr_a 3e-5" +arg --seed 0 1 +no-tag-arg --seed
onager prelaunch --exclude-job-id +jobname door_gs1_6_HER_3e-5_3e-3 +command "python -m hrl --experiment_name door_gs1_6_HER_3e-5_3e-3 --results_dir ~/scratch/results --device 'cuda:0' --environment door --episodes 20000 --sample_method random --use_HER --lr_c 3e-5 --lr_a 3e-3" +arg --seed 0 1 +no-tag-arg --seed
onager prelaunch --exclude-job-id +jobname door_gs1_7_HER_3e-5_3e-4 +command "python -m hrl --experiment_name door_gs1_7_HER_3e-5_3e-4 --results_dir ~/scratch/results --device 'cuda:0' --environment door --episodes 20000 --sample_method random --use_HER --lr_c 3e-5 --lr_a 3e-4" +arg --seed 0 1 +no-tag-arg --seed
onager prelaunch --exclude-job-id +jobname door_gs1_8_HER_3e-5_3e-5 +command "python -m hrl --experiment_name door_gs1_8_HER_3e-5_3e-5 --results_dir ~/scratch/results --device 'cuda:0' --environment door --episodes 20000 --sample_method random --use_HER --lr_c 3e-5 --lr_a 3e-5" +arg --seed 0 1 +no-tag-arg --seed

onager prelaunch --exclude-job-id +jobname door_gs1_9_3e-3_3e-3 +command "python -m hrl --experiment_name door_gs1_9_3e-3_3e-3 --results_dir ~/scratch/results --device 'cuda:0' --environment door --episodes 20000 --sample_method random --lr_c 3e-3 --lr_a 3e-3" +arg --seed 0 1 +no-tag-arg --seed
onager prelaunch --exclude-job-id +jobname door_gs1_10_3e-3_3e-4 +command "python -m hrl --experiment_name door_gs1_10_3e-3_3e-4 --results_dir ~/scratch/results --device 'cuda:0' --environment door --episodes 20000 --sample_method random --lr_c 3e-3 --lr_a 3e-4" +arg --seed 0 1 +no-tag-arg --seed
onager prelaunch --exclude-job-id +jobname door_gs1_11_3e-3_3e-5 +command "python -m hrl --experiment_name door_gs1_11_3e-3_3e-5 --results_dir ~/scratch/results --device 'cuda:0' --environment door --episodes 20000 --sample_method random --lr_c 3e-3 --lr_a 3e-5" +arg --seed 0 1 +no-tag-arg --seed
onager prelaunch --exclude-job-id +jobname door_gs1_12_3e-4_3e-3 +command "python -m hrl --experiment_name door_gs1_12_3e-4_3e-3 --results_dir ~/scratch/results --device 'cuda:0' --environment door --episodes 20000 --sample_method random --lr_c 3e-4 --lr_a 3e-3" +arg --seed 0 1 +no-tag-arg --seed
onager prelaunch --exclude-job-id +jobname door_gs1_13_3e-4_3e-4 +command "python -m hrl --experiment_name door_gs1_13_3e-4_3e-4 --results_dir ~/scratch/results --device 'cuda:0' --environment door --episodes 20000 --sample_method random --lr_c 3e-4 --lr_a 3e-4" +arg --seed 0 1 +no-tag-arg --seed
onager prelaunch --exclude-job-id +jobname door_gs1_14_3e-4_3e-5 +command "python -m hrl --experiment_name door_gs1_14_3e-4_3e-5 --results_dir ~/scratch/results --device 'cuda:0' --environment door --episodes 20000 --sample_method random --lr_c 3e-4 --lr_a 3e-5" +arg --seed 0 1 +no-tag-arg --seed
onager prelaunch --exclude-job-id +jobname door_gs1_15_3e-5_3e-3 +command "python -m hrl --experiment_name door_gs1_15_3e-5_3e-3 --results_dir ~/scratch/results --device 'cuda:0' --environment door --episodes 20000 --sample_method random --lr_c 3e-5 --lr_a 3e-3" +arg --seed 0 1 +no-tag-arg --seed
onager prelaunch --exclude-job-id +jobname door_gs1_16_3e-5_3e-4 +command "python -m hrl --experiment_name door_gs1_16_3e-5_3e-4 --results_dir ~/scratch/results --device 'cuda:0' --environment door --episodes 20000 --sample_method random --lr_c 3e-5 --lr_a 3e-4" +arg --seed 0 1 +no-tag-arg --seed
onager prelaunch --exclude-job-id +jobname door_gs1_17_3e-5_3e-5 +command "python -m hrl --experiment_name door_gs1_17_3e-5_3e-5 --results_dir ~/scratch/results --device 'cuda:0' --environment door --episodes 20000 --sample_method random --lr_c 3e-5 --lr_a 3e-5" +arg --seed 0 1 +no-tag-arg --seed

onager launch --backend slurm --jobname door_gs1_0_HER_3e-3_3e-3 --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
onager launch --backend slurm --jobname door_gs1_1_HER_3e-3_3e-4 --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
onager launch --backend slurm --jobname door_gs1_2_HER_3e-3_3e-5 --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
onager launch --backend slurm --jobname door_gs1_3_HER_3e-4_3e-3 --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
onager launch --backend slurm --jobname door_gs1_4_HER_3e-4_3e-4 --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
onager launch --backend slurm --jobname door_gs1_5_HER_3e-4_3e-5 --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
onager launch --backend slurm --jobname door_gs1_6_HER_3e-5_3e-3 --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
onager launch --backend slurm --jobname door_gs1_7_HER_3e-5_3e-4 --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
onager launch --backend slurm --jobname door_gs1_8_HER_3e-5_3e-5 --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
onager launch --backend slurm --jobname door_gs1_9_3e-3_3e-3 --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
onager launch --backend slurm --jobname door_gs1_10_3e-3_3e-4 --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
onager launch --backend slurm --jobname door_gs1_11_3e-3_3e-5 --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
onager launch --backend slurm --jobname door_gs1_12_3e-4_3e-3 --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
onager launch --backend slurm --jobname door_gs1_13_3e-4_3e-4 --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
onager launch --backend slurm --jobname door_gs1_14_3e-4_3e-5 --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
onager launch --backend slurm --jobname door_gs1_15_3e-5_3e-3 --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
onager launch --backend slurm --jobname door_gs1_16_3e-5_3e-4 --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
onager launch --backend slurm --jobname door_gs1_17_3e-5_3e-5 --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
```
```
onager prelaunch --exclude-job-id +jobname door_trial_1_her_clf +command "python -m hrl --experiment_name door_trial_1_her_clf --results_dir ~/scratch/results --device 'cuda:0' --environment door --episodes 20000 --sample_method classifier --use_HER --lr_c 3e-5 --lr_a 3e-5" +arg --seed 0 1 2 3 4 +no-tag-arg --seed
onager prelaunch --exclude-job-id +jobname door_trial_1_her_uw_clf +command "python -m hrl --experiment_name door_trial_1_her_uw_clf --results_dir ~/scratch/results --device 'cuda:0' --environment door --episodes 20000 --sample_method classifier_unweighted --use_HER --lr_c 3e-5 --lr_a 3e-5" +arg --seed 0 1 2 3 4 +no-tag-arg --seed
onager prelaunch --exclude-job-id +jobname door_trial_1_her_oracle +command "python -m hrl --experiment_name door_trial_1_her_oracle --results_dir ~/scratch/results --device 'cuda:0' --environment door --episodes 20000 --sample_method oracle --use_HER --lr_c 3e-5 --lr_a 3e-5" +arg --seed 0 1 2 3 4 +no-tag-arg --seed
onager prelaunch --exclude-job-id +jobname door_trial_1_her_baseline +command "python -m hrl --experiment_name door_trial_1_her_baseline --results_dir ~/scratch/results --device 'cuda:0' --environment door --episodes 20000 --sample_method random --use_HER --lr_c 3e-5 --lr_a 3e-5" +arg --seed 0 1 2 3 4 +no-tag-arg --seed

onager launch --backend slurm --jobname door_trial_1_her_clf --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
onager launch --backend slurm --jobname door_trial_1_her_uw_clf --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
onager launch --backend slurm --jobname door_trial_1_her_oracle --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
onager launch --backend slurm --jobname door_trial_1_her_baseline --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
```

```
onager prelaunch --exclude-job-id +jobname switch_trial_2_her_clf +command "python -m hrl --experiment_name switch_trial_2_her_clf --results_dir ~/scratch/results --device 'cuda:0' --environment switch --episodes 20000 --sample_method classifier --use_HER --lr_c 3e-6 --lr_a 3e-6" +arg --seed 0 1 2 3 4 +no-tag-arg --seed
onager prelaunch --exclude-job-id +jobname switch_trial_2_her_uw_clf +command "python -m hrl --experiment_name switch_trial_2_her_uw_clf --results_dir ~/scratch/results --device 'cuda:0' --environment switch --episodes 20000 --sample_method classifier_unweighted --use_HER --lr_c 3e-6 --lr_a 3e-6" +arg --seed 0 1 2 3 4 +no-tag-arg --seed
onager prelaunch --exclude-job-id +jobname switch_trial_2_her_oracle +command "python -m hrl --experiment_name switch_trial_2_her_oracle --results_dir ~/scratch/results --device 'cuda:0' --environment switch --episodes 20000 --sample_method oracle --use_HER --lr_c 3e-6 --lr_a 3e-6" +arg --seed 0 1 2 3 4 +no-tag-arg --seed
onager prelaunch --exclude-job-id +jobname switch_trial_2_her_baseline +command "python -m hrl --experiment_name switch_trial_2_her_baseline --results_dir ~/scratch/results --device 'cuda:0' --environment switch --episodes 20000 --sample_method random --use_HER --lr_c 3e-6 --lr_a 3e-6" +arg --seed 0 1 2 3 4 +no-tag-arg --seed

onager launch --backend slurm --jobname switch_trial_2_her_clf --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
onager launch --backend slurm --jobname switch_trial_2_her_uw_clf --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
onager launch --backend slurm --jobname switch_trial_2_her_oracle --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
onager launch --backend slurm --jobname switch_trial_2_her_baseline --duration 1-23:59:59 --gpus 1 --cpus 4 --mem 44 --partition 3090-gcondo
```