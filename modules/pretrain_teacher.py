import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from models.teacher_network import TeacherNetwork, knn_graph
from models.decoder_network import DecoderNetwork
from modules.data.synthetic_dataset import ModelNetDataset
from modules.data.modelnet10_data import SYNTHETIC_DATA_PATH


NUM_FEATURE_SAMPLES = 16
device = torch.device('cuda:0')

train_dataset = ModelNetDataset(root_dir=SYNTHETIC_DATA_PATH / 'train')
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

test_dataset = ModelNetDataset(root_dir=SYNTHETIC_DATA_PATH / 'test')
test_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)


class ChamferDistanceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, point_cloud1: torch.Tensor, point_cloud2: torch.Tensor):
        # Compute pairwise distance matrix between point_cloud1 and point_cloud2
        diff = point_cloud1[:, None, :] - point_cloud2[None, :, :]
        dist_matrix = torch.sum(diff ** 2, dim=-1)

        min_dist1, _ = torch.min(dist_matrix, dim=1)
        min_dist2, _ = torch.min(dist_matrix, dim=0)

        chamfer_dist = torch.mean(min_dist1) + torch.mean(min_dist2)

        return chamfer_dist


def get_rf_and_average(knn: torch.Tensor, point_cloud: torch.Tensor, center_idx: int, N: int):
    # Initialize variables to keep track of the sum of coordinates and the count of neighbors
    sum_coords = torch.zeros(3, dtype=torch.float32).to(device)
    neighbor_count = 0

    # Initialize a set to keep track of visited nodes
    visited = set()
    # Initialize a queue with the starting node and its initial distance (0)
    queue = [(center_idx, 0)]
    # Mark the starting node as visited
    visited.add(center_idx)

    while queue:
        current_node, current_distance = queue.pop(0)

        # If the current distance is less than N, add neighbors to the queue
        if current_distance < N:
            neighbors = knn[current_node]
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, current_distance + 1))

                    # Add the neighbor's coordinates to the running sum and increment the count
                    sum_coords += point_cloud[neighbor]
                    neighbor_count += 1

    # Calculate the average coordinates
    if neighbor_count > 0:
        average_coords = sum_coords / neighbor_count
    else:
        # If no neighbors are found within N steps, set the average to a zero tensor
        average_coords = torch.zeros(3, dtype=torch.float32).to(device)

    # Convert the visited set to a sorted list and then to a tensor, excluding the start node
    visited.remove(center_idx)
    points_tensor = torch.tensor(sorted(visited), dtype=torch.long).to(device)

    return points_tensor, average_coords


def pretrain_teacher_network(teacher_net, decoder_net, dataset, num_points, num_blocks, k, device, d_model):
    teacher_net = teacher_net.to(device)
    decoder_net = decoder_net.to(device)
    criterion = ChamferDistanceLoss().to(device)
    optimizer = torch.optim.Adam(list(teacher_net.parameters()) + list(decoder_net.parameters()), lr=1e-3)

    for data in tqdm(dataset, desc='Training'):
        point_cloud = data['scene'][0].to(device)
        del data
        
        knn = knn_graph(point_cloud, k)

        descriptors = teacher_net(point_cloud, knn)

        sampled_indices = torch.randint(0, len(point_cloud), (NUM_FEATURE_SAMPLES,))

        total_loss = 0.0
        for i in tqdm(sampled_indices, 'Calculating loss for reconstructed points', leave=False):
            reconstructed_points = decoder_net(descriptors[i]).reshape(-1, 3)

            # 4 resnet blocks with 2 LFA blocks each
            rf_indices, mean_point = get_rf_and_average(knn, point_cloud, i, num_blocks * 2)
            rf_centered = (point_cloud[rf_indices] - mean_point)

            loss = criterion(reconstructed_points, rf_centered) / NUM_FEATURE_SAMPLES
            total_loss += loss.item()

        print(total_loss)
        total_loss = torch.tensor(total_loss, requires_grad=True)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return teacher_net, decoder_net


if __name__ == '__main__':
    # Example usage
    d_model = 64
    k = 5
    num_blocks = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    t_net = TeacherNetwork(d_model, k, 16000, num_blocks, device=device).to(device)
    decoder = DecoderNetwork(d_model, 32).to(device)

    pretrained_teacher_net, pretrained_decoder_net = pretrain_teacher_network(t_net, decoder, train_dataloader, 16000, num_blocks, k, device, d_model)
