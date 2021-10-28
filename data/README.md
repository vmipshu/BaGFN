# Data Preprocess

- Thanks to [AutoInt](https://github.com/DeepGraphLearning/RecommenderSystems/tree/master/featureRec). 
  We employ the data process from this repository.

## Input Format

BaGFN requires the input data in the following format:

- train_x: matrix with shape (num_sample, num_field). 
  train_x[s][t] is the feature value of feature field t of sample s in the dataset. 
  The default value for categorical feature is 1.
- train_i: matrix with shape (num_sample, num_field). 
  train_i[s][t] is the feature index of feature field t of sample s in the dataset. 
  The maximal value of train_i is the feature size.
- train_y: label of each sample in the dataset.

If you want to know how to preprocess the data, please refer to Dataprocess/Criteo/preprocess.py

## Example

We use two public real-world datasets(Avazu, Criteo) in our experiments. 
Since the two datasets are super huge, they can not be fit into the memory as a whole. 
In our implementation, we split the whole dataset into 10 parts and 
we use the first file as test set and the second file as valid set. 
We provide the codes for preprocessing these two datasets in the directory `Dataprocess/`. 
If you want to reuse these codes, you should first run `preprocess.py` to generate
train_x.txt, train_i.txt, train_y.txt as described in Input Format. 
Then you should run `Dataprocess/Kfold_split/StratifiedKfold.py` to split the whole dataset into ten folds. 
Finally you can run scale.py to scale the numerical value(optional).

To help test the correctness of the code and familarize yourself with the code, 
we upload the first 10000 samples of Criteo dataset in train_examples.txt. 
And we provide the scripts for preprocessing and training. 
(Please refer to `sample_preprocess.sh`, 
you may need to modify the path in `Dataprocess/Criteo/config.py` and `sample_preprocess.sh`).

After you run the `sample_preprocess.sh`, 
you should get a folder named Criteo which contains part*, feature_size.npy, fold_index.npy, train_*.txt. 
feature_size.npy contains the number of total features which will be used to initialize the model. 
train_*.txt is the whole dataset.

Here's how to run the preprocessing.

```shell
cd data
mkdir Criteo
python ./Dataprocess/Criteo/preprocess.py
python ./Dataprocess/Kfold_split/stratifiedKfold.py
python ./Dataprocess/Criteo/scale.py
```