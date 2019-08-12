
.PHONY: data



data:
	kaggle competitions download -c histopathologic-cancer-detection -p data
	unzip data/train.zip -d data/train
	unzip data/train_labels.csv.zip -d data/
	chmod 777 data/train_labels.csv # weird permission issue
	rm data/*.zip # cleanup


