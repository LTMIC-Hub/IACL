<div align="center">

<h1>Implicit Augmentation Contrastive Learning for long-tailed medical image classification</h1>

</div>


## Requirements
- Linux both support
- python 3.8
- PyTorch 1.13.1
- torchvision 0.10.0

## Training
To train the model(s) in the paper, run this command:
```python
python train.py --dataset ISIC2018 --data_path /home/Data/311/ISIC_2018_DataSet-311/ --batch_size 64 --lr 0.002 --epochs 100 --gpu 0 --model_path ./results/ISIC-2018/
python train.py --dataset ISIC2019 --data_path /home/Data/311/ISIC_2019_DataSet-311/ --batch_size 64 --lr 0.002 --epochs 100 --gpu 0 --model_path ./results/ISIC-2019/
```

## Testing
```python
python test.py --dataset ISIC2019 --data_path /home/Data/311/ISIC_2019_DataSet-311/  --model_path ./results/ISIC-2019/model_*.pth --log_path ./test.txt
python test.py --dataset ISIC2018 --data_path /home/Data/311/ISIC_2018_DataSet-311/  --model_path ./results/ISIC-2018/model_*.pth --log_path ./test.txt
```

