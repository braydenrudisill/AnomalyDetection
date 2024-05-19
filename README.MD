# Pivot Robotics Computer Vision Challenge

Implementation of the 3D Student-Teacher (3D-ST) method for 
anomaly detection of 3D data as described in 
[this paper](https://arxiv.org/pdf/2202.11660).

## Synthetic Dataset Generation
A synthetic dataset is generated using the method described
in page 9 of the paper consisting of a training and validation
set of 500 and 25 point clouds using the ModelNet10 dataset.
The number of points sampled per point cloud is set to 16,000 
instead of the 64,000 as in the original paper to save time.


```bash
python -m modules.
```

## Teacher and Decoder Pretraining
Because of the smaller point_cloud size, k is reduced to 8
from 32 and m is reduced to 128.  The feature vectors are
64-dimensional and there are 4 residual network blocks.
The models were trained for 250 epochs, where at each time 
step, 16 random feature vectors are sampled from the teacher's
output and fed to the decoder. The total chamfer distance
between the decoder's output clouds and the receptive field
clouds is used as the loss for both models.