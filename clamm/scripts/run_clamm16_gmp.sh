pipenv run python3 clamm.py experiments/clamm16_gmp\
	-c configs/clamm16_cluster.conf \
	--gmp_lambda 5000 \
	--config_optim configs/adam.conf \
	--pool_type gmp \
	--model_name poolnet 
