# Train
python train_diffusion.py \
  --train_npz data/processed/modelnet40_completion_official/train.npz \
  --val_npz data/processed/modelnet40_completion_official/val.npz \
  --workdir runs/diffusion_baseline_500 \
  --epochs 500 \
  --batch_size 512 \
  --lr 2e-4 \
  --num_steps 1000


# Train RL
python train_rl_scheduler.py \
  --ckpt runs/diffusion_baseline_500/checkpoints/best.pt \
  --train_npz data/processed/modelnet40_completion_official/train.npz \
  --val_npz data/processed/modelnet40_completion_official/val.npz \
  --workdir runs/rl_scheduler_500_10000 \
  --episodes 10000 \
  --lambda_steps 0.002 \
  --max_steps 500 \
  --num_steps 1000 \
  --actions 1 2 4 \
  --lr 1e-4 \
  --log_interval 50 \
  --eval_interval 200

# Evaluation
python eval_rl_scheduler.py \
  --ckpt runs/diffusion_baseline_500/checkpoints/best.pt \
  --policy_ckpt runs/rl_scheduler_500_10000/checkpoints/best_policy.pt \
  --test_npz data/processed/modelnet40_completion_official/test.npz \
  --num_steps 1000 \
  --actions 1 2 4 \
  --lambda_steps 0.002 \
  --max_steps 500