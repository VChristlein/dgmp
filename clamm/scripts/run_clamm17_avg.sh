pipenv run python3 clamm.py experiments/clamm17_avg\
        -c configs/clamm17.conf \
	--config_optim configs/adam.conf \
        --pool_type avg \
        --model_name resnet50 \
	--start_exp_decay 16 

