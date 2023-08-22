import time

# Comment this out because we want to override it from commandline args
# out_dir = 'out-shakespeare'
eval_interval = 5
eval_iters = 40
wandb_log = False # feel free to turn on
wandb_project = 'shakespeare'
wandb_run_name = 'ft-' + str(time.time())

# Comment this out because we want to override it from commandline args
# dataset = 'shakespeare'
# init_from = 'gpt2-xl' # this is the largest GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 20

# finetune at constant LR
# NOTE(yiming) Don't go too high here or it will overfit our tiny dataset 
# Do play around with this if you can  
learning_rate = 4e-5
decay_lr = False
