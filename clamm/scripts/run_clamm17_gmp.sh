pipenv run python3 clamm.py experiments/clamm17_gmp\
	-c configs/clamm17_cluster.conf \
	--gmp_lambda 5000 \
	--config_optim configs/adam.conf \
	--pool_type gmp \
	--model_name poolnet \
	--start_exp_decay 16 
