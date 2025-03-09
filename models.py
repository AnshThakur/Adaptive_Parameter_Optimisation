import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskCNN(nn.Module):
    def __init__(self, output_sizes):
        super(MultiTaskCNN, self).__init__()
        # Shared layers (Convolutional + Pooling)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected shared layer before task-specific layers
        self.fc_shared = nn.Linear(64 * 7 * 7, 128)  # 7x7 is the size after pooling
        
        # Task-specific output layers
        self.task_layers = nn.ModuleList([nn.Linear(128, out_size) for out_size in output_sizes])
    
    def forward(self, x, task_id):
        # Shared convolutional layers
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        
        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Shared fully connected layer
        x = torch.relu(self.fc_shared(x))
        
        # Task-specific output
        return self.task_layers[task_id](x)



class MoEBlock(nn.Module):
    def __init__(self, num_experts, input_size, output_size, k=4):
        super(MoEBlock, self).__init__()
        self.num_experts = num_experts
        self.k = k
        self.experts = nn.ModuleList([nn.Linear(input_size, output_size) for _ in range(num_experts)])
        self.routing_network = nn.Linear(input_size, num_experts)

    def forward(self, x):
        # Get routing weights
        routing_weights = F.softmax(self.routing_network(x), dim=-1)

        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.k, dim=-1)

    
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # Shape: [batch_size, num_experts, output_size]

        # Gather the top-k expert outputs for each input
        top_k_expert_outputs = torch.gather(expert_outputs, 1, top_k_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.size(-1)))

        # Weight and sum the top-k expert outputs
        top_k_weights = top_k_weights.unsqueeze(-1)  # Shape: [batch_size, k, 1]
        output = torch.sum(top_k_weights * top_k_expert_outputs, dim=1)  # Weighted sum of expert outputs

        return output


class ModSquadCNN(nn.Module):
    def __init__(self, output_sizes, num_experts=4,k=2):
        super(ModSquadCNN, self).__init__()
        # Shared layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # MoE Block
        self.moe = MoEBlock(num_experts, 64 * 7 * 7, 128,k=k)
        
        # Task-specific output layers
        self.task_layers = nn.ModuleList([nn.Linear(128, out_size) for out_size in output_sizes])

    def forward(self, x, task_id):
        # Shared layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        
        # MoE Block
        x = self.moe(x)
        
        # Task-specific output
        return self.task_layers[task_id](x)
