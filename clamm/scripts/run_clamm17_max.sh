pipenv run python3 clamm.py experiments/clamm17_max\
	-c configs/clamm17_cluster.conf \
	--config_optim configs/adam.conf \
	--pool_type max \
	--model_name resnet50 \
	--start_exp_decay 16  

