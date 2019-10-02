if [ -z "$1" ]
then
	echo "\$1 is empty. Set name for zipfile"
else
	zip $1 spatial_monet/model.py spatial_monet/main.py spatial_monet/experiment_config.py
	python main.py --visdom_env $1 --load_location ../monet_checkpoints/$1 --load_params
fi
