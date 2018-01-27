# PointNet3

## Setup
```sh
virtualenv -p python3 env_name
python setup.py
```

## Training
```sh
python train.py  --model {lstm/lstm_dist, default:lstm} specify which model to run
                 --learning_rate {float, default:0.01} learning rate for the model
                 --batchSize {int, default:32} batch size for dataset
                 --nepoch {int, default:100} number of epoch to run
                 --sort {0/1, default:0} if 1, input sequences will be sorted
                 --distance {0/1, default:0} NOT YET IMPLEMENTED, if 1, will return 6-dim tensor show distance from next point
                 --num_points {256/512/1024/2048, default:2048} NOT YET IMPLEMENTED number of points in input sequence
                 --workers {int, default:4} number of workers for PyTorch dataloader
                 --outf {str, default:'logs'} output path of loggings
                 --path {str, default:''} path to load pre-trained model for transfer learning
                 --debug {0/1, default:0} if 1, no folder/logging will be recorded
``` 
## Visualize loss/accuracy graph
**use python2**
tensorboard --logdir ./logs/{DATE:TIME}/
