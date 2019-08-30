if [ -z "$1" ]
then
	echo "\$1 is empty. Set name for zipfile"
else
	zip $1 model.py main.py experiment_config.py
	python main.py --constrain_theta --visdom_env $1
fi
