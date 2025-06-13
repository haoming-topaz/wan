accelerate launch train_wanseg.py \
   --set_grads_to_none \
   --allow_tf32 \
   --target_size 512 512 \
   --multi_step_maximum 1
   # --resume_from_checkpoint "/home/topaz/haoming/seg/exp-wan/25-06-12-17-44/checkpoints/model_5005.pth"
