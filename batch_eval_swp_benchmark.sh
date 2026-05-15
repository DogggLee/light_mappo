python scripts/batch_eval_best_models.py --root_dir $1 --task_file config/eval_tasks/ablation/baseline_v2.json --model_glob best_eval_capture_rate --cuda &&

python scripts/batch_eval_best_models.py --root_dir $1 --task_file config/eval_tasks/ablation/hard_b100.json --model_glob best_eval_capture_rate --cuda  &&

python scripts/batch_eval_best_models.py --root_dir $1 --task_file config/eval_tasks/ablation/full_b100.json --model_glob best_eval_capture_rate --cuda &&

python scripts/batch_eval_best_models.py --root_dir $1 --task_file config/eval_tasks/ablation/easy_b100.json --model_glob best_eval_capture_rate --cuda
