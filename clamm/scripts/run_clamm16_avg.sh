pipenv run python3 clamm.py experiments/clamm16_avg \
        -c configs/clamm16.conf \
	--config_optim configs/adam.conf \
        --pool_type avg \
        --model_name resnet50 

