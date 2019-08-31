if [ -z "$1" ]
then
	echo "\$1 is empty. Set name for zipfile"
else
	zip $1 src/model.py src/main.py src/experiment_config.py
	python src/main.py --constrain_theta --visdom_env $1
fi
