import torch


class EuclidianFeatureDistance(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, student_features, teacher_features, means, std_devs):
        """Takes the average feature-wise L2 difference between the features from the two point clouds.

        Student_features: (n, d)
        teacher_features: (n, d)
        means: (d,)
        std_devs: (d,)
        """
        total_distance = torch.sum((student_features - (teacher_features - means) / std_devs).pow(2).sum(dim=1).sqrt())

        return total_distance / len(teacher_features)
