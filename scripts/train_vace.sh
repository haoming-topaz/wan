accelerate launch train_wanseg.py \
   --set_grads_to_none \
   --allow_tf32 \
   --target_size 512 512 \
   --multi_step_maximum 1
