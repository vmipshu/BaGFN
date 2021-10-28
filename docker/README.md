# BaGFN: Broad Attentive Graph Fusion Network for High-Order Feature Interactions

## Docker environment setting and training.

1. Build a docker image: 
```shell
cd docker
docker build -it bagfn:v1 .
```
2. Run a docker container using the bagfn:v1 image:
```shell
docker run -it --rm --gpus all bagfn:v1 bash
```
3. Process the train_examples.txt:
```shell
root@xxxx:/workspace/BaGFN# cd data
root@xxxx:/workspace/BaGFN/data# bash sample_preprocess.sh
```
4. Train the model:
```shell
root@xxxx:/workspace/BaGFN/data# cd ..
root@xxxx:/workspace/BaGFN# python train.py --batch_size 4
```
5. We can get log and checkpoint files in `/workspace/BaGFN/logs/Criteo` and `/workspace/BaGFN/checkpoints/Criteo` as default.  
