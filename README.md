# FinalYearProject
 
# FinalYearProject
 
This project builds upon the research done by [Meidan et. Al](https://arxiv.org/pdf/1805.03409.pdf) and uses the [N-BaloT](https://www.kaggle.com/datasets/mkashifn/nbaiot-dataset) dataset which they created from their experiments. The title of this project is 'Exploring the effectiveness of transfer learning on autoencoders for botnet attack detection.' This project tries to answer whether or not a Machine Learning technique called Trasfer Learning is useful for the development of IoT intrusion detection systems, specifically for botnet attacks.

The submission branch of this repository uses a subset of the N-BaloT dataset. It is also the brach that has the most development done on it and so should be the branch used when running this project.

## Setup
1. clone the repository
```bash
git clone --branch submission https://github.com/ZafirProjects/FinalYearProject.git
```
2. Download the N-BaloT dataset
3. Unzip it and put the folder into the cloned project
4. Seperate the following files from the dataset folder (optionally put it in a folder called metadataset):
   data_summary.csv
   device_info.csv
   features.csv
   README.md
5. Add the following folders (might be created automatically by running the program)
   logs/fit
   metrics
   neuralnetworks
   statistics
   
Your project should look like this:
/dataset      (should contain the N-Balot dataset without the metadata)
/logs
/metadaset    (contains the metadata for the N-BaloT dataset)
/metrics
/neuralnetworks
/statistics
conv_autoencoder_attempt.py (useless)
deep_autoencoder.py
metric_processor.py
tensorboard.ipynb
transfer_learning_autoencoder.py

## Running the project
Your best bet would be to run the following files in order:
1. deep_autoencoder.py
2. transfer_learning_autoencoder.py
3. tensorboard.ipynb
4. metric_processor.py
