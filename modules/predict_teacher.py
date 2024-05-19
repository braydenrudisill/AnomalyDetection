import torch

from modules.data.modelnet10_data import M10_SYNTHETIC_16k
from modules.pretrain_teacher import get_rf
from modules.models import TeacherNetwork, DecoderNetwork, KNNGraph


def main():
    d_model = 64
    k = 8
    num_blocks = 4
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

    teacher = TeacherNetwork(d_model, k, num_blocks, device=device).to(device)
    decoder = DecoderNetwork(d_model, 128).to(device)
    knn_graph = KNNGraph()

    teacher.load_state_dict(torch.load('/modules/models/teacher.pt'))
    decoder.load_state_dict(torch.load('/modules/models/decoder.pt'))

    with open(M10_SYNTHETIC_16k / 'test/12.txt', 'r') as f:
        sample_point_cloud = torch.tensor([[float(c) for c in line.strip().split(' ')] for line in f], dtype=torch.float) / 0.015

    knn = knn_graph(sample_point_cloud, 8)
    features = teacher(sample_point_cloud.to(device))
    print(features)
    rf_indices = get_rf(knn, torch.tensor([0]), 4 * 2)
    rf = sample_point_cloud[rf_indices]
    rf_centered = rf - torch.mean(rf, dim=0)
    pred_cloud = decoder(features[0]).reshape(-1,3)
    print(pred_cloud[:10])

    print('Writing to file.')
    with open('predicted_cloud_12_0.txt', 'w+') as f:
        f.writelines(' '.join(str(c.item()) for c in pt) + '\n' for pt in pred_cloud)

    with open('rf_12_0.txt', 'w+') as f:
        f.writelines(' '.join(str(c.item()) for c in pt) + '\n' for pt in rf_centered)

    print('Done writing.')


if __name__ == '__main__':
    main()
