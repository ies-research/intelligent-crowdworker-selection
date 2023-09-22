#!/usr/bin/env bash
#SBATCH --job-name=letter
#SBATCH --array=1-3%3
#SBATCH --mem=20gb
#SBATCH --ntasks=1
#SBATCH --get-user-env
#SBATCH --cpus-per-task=4
#SBATCH --partition=main
#SBATCH --output=/path/to/be/specified/letter_%A_%a.log
#SBATCH --gres=gpu:1
eval "$(sed -n "$(($SLURM_ARRAY_TASK_ID+13)) p" run_experiment_letter.sh)"
exit 0

srun python -u /path/to/be/specified/intelligent_crowdworker_selection/evaluation/run_experiment.py with seed=0 data_set_name=letter data_type=none n_repeats=5 test_size=0.2 n_al_cycles=25 al_batch_size=256 initial_label_size=16 annot_perf_sel=False trainer_dict.max_epochs=100 trainer_dict.enable_progress_bar=False trainer_dict.enable_checkpointing=False trainer_dict.accelerator=gpu trainer_dict.devices=1 trainer_dict.logger=False optimizer=AdamW optimizer_dict.lr=0.01 optimizer_dict.weight_decay=0.0 lr_scheduler=CosineAnnealing lr_scheduler_dict.T_max=100 batch_size=64 dropout_rate=0.0 model_name=mr model_dict.confusion_matrix=isotropic model_dict.embed_size=64 model_dict.embed_x=none model_dict.use_annotator_features=False model_dict.ap_use_residual=False model_dict.ap_use_outer_product=False
srun python -u /path/to/be/specified/intelligent_crowdworker_selection/evaluation/run_experiment.py with seed=0 data_set_name=letter data_type=none n_repeats=5 test_size=0.2 n_al_cycles=25 al_batch_size=256 initial_label_size=16 annot_perf_sel=True trainer_dict.max_epochs=100 trainer_dict.enable_progress_bar=False trainer_dict.enable_checkpointing=False trainer_dict.accelerator=gpu trainer_dict.devices=1 trainer_dict.logger=False optimizer=AdamW optimizer_dict.lr=0.01 optimizer_dict.weight_decay=0.0 lr_scheduler=CosineAnnealing lr_scheduler_dict.T_max=100 batch_size=64 dropout_rate=0.0 model_name=madl model_dict.eta=0.8 model_dict.confusion_matrix=full model_dict.alpha=1.25 model_dict.beta=0.25 model_dict.embed_size=64 model_dict.embed_x=learned model_dict.use_annotator_features=False model_dict.ap_use_residual=True model_dict.ap_use_outer_product=True
srun python -u /path/to/be/specified/intelligent_crowdworker_selection/evaluation/run_experiment.py with seed=0 data_set_name=letter data_type=none n_repeats=5 test_size=0.2 n_al_cycles=25 al_batch_size=256 initial_label_size=16 annot_perf_sel=False trainer_dict.max_epochs=100 trainer_dict.enable_progress_bar=False trainer_dict.enable_checkpointing=False trainer_dict.accelerator=gpu trainer_dict.devices=1 trainer_dict.logger=False optimizer=AdamW optimizer_dict.lr=0.01 optimizer_dict.weight_decay=0.0 lr_scheduler=CosineAnnealing lr_scheduler_dict.T_max=100 batch_size=64 dropout_rate=0.0 model_name=madl model_dict.eta=0.8 model_dict.confusion_matrix=full model_dict.alpha=1.25 model_dict.beta=0.25 model_dict.embed_size=64 model_dict.embed_x=learned model_dict.use_annotator_features=False model_dict.ap_use_residual=True model_dict.ap_use_outer_product=True