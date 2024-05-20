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
python -m modules.data.create_datasets
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

```bash
python -m modules.pretrain_teacher
```

## Student Training
We train the student model using the same architecture as the 
teacher. The loss is calculated via the average L2 distance
between the student's and teacher's feature vectors. Since
the student is only being trained on anomaly-free data, the
idea is that it will only learn to predict features that are
anomaly free, and the error from the teacher can be measured
as a form of heatmap for defects.

```bash
python -m modules.train_student
```

## Inference

```bash
python -m modules.create_heatmap
```

## Examples
### Crack anomalies
![example 1](https://github.com/braydenrudisill/AnomalyDetection/assets/55212800/ba51e37f-714b-4a22-b1ca-c9109769d949)
![example 2](https://github.com/braydenrudisill/AnomalyDetection/assets/55212800/b6296479-6520-4a36-94ef-5b49d8fe72e9)
![example 3](https://github.com/braydenrudisill/AnomalyDetection/assets/55212800/1c2530bc-9291-4bd0-a027-d27411779fc1)
![example 4](https://github.com/braydenrudisill/AnomalyDetection/assets/55212800/63ae53fc-d734-4a50-b8d9-e37b51cae784)

### Hole defect
![example 5](https://github.com/braydenrudisill/AnomalyDetection/assets/55212800/8a150eed-188c-4a1d-b8c8-b8e8207501a7)

### Anomaly-free
![no anomaly](https://github.com/braydenrudisill/AnomalyDetection/assets/55212800/329f0432-910d-4a75-b9af-2daed311bc3c)


### Teacher training loss
![teacher_network training loss plot](https://github.com/braydenrudisill/AnomalyDetection/assets/55212800/9e0266a3-14db-446f-ae8c-7f6a7b9b0649)

### Student network training loss
![student_network training loss plot](https://github.com/braydenrudisill/AnomalyDetection/assets/55212800/afdd304b-4ffb-41c0-8150-5ecc2f8ad709)
