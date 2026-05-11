python scripts/batch_eval_best_models.py --root_dir results_old/plots/Tech --task_file config/eval_tasks/ablation/baseline_v2.json --model_glob best_eval_capture_rate --cuda &&

python scripts/batch_eval_best_models.py --root_dir results_old/plots/Tech --task_file config/eval_tasks/ablation/base_capdis.json --model_glob best_eval_capture_rate --cuda  &&

python scripts/batch_eval_best_models.py --root_dir results_old/plots/Tech --task_file config/eval_tasks/ablation/base_ms_v2.json --model_glob best_eval_capture_rate --cuda &&

python scripts/batch_eval_best_models.py --root_dir results_old/plots/Tech --task_file config/eval_tasks/ablation/base_ws_v2.json --model_glob best_eval_capture_rate --cuda
