import torch
import torch.optim as optim

class AdaptiveProtectiveOptimizer(optim.SGD):
    def __init__(self, model, task_specific_layer_names, num_tasks, lr=0.05, k=0.5,protect_factor=0.005, initial_threshold=1e-4,beta=0.9):
        super(AdaptiveProtectiveOptimizer, self).__init__(model.parameters(), lr=lr)
        self.protect_factor = protect_factor
        self.initial_threshold = initial_threshold
        self.adaptive_thresholds = {}  # Store layer-wise thresholds
        self.num_tasks = num_tasks
        self.mu=1
        self.beta=beta
        self.k=k

        # Exclude task-specific layers from tracking by their names
        self.shared_params = {name: p for name, p in model.named_parameters() if name not in task_specific_layer_names}
        self.initial_params = {name: p.clone().detach() for name, p in self.shared_params.items()}
 
        # Global binary mask for tracking updated parameters
        self.global_mask = {name: torch.zeros_like(p, dtype=torch.bool) for name, p in self.shared_params.items()}

        # Task-specific masks for each task
        self.task_masks = {task_id: {name: torch.zeros_like(p, dtype=torch.bool) for name, p in self.shared_params.items()} for task_id in range(self.num_tasks)}

        # History of gradients used for adaptive threshold updates
        self.param_history = {name: torch.zeros_like(p) for name, p in self.shared_params.items()}
        
        # Initialize layer-wise adaptive thresholds with the same initial value
        for name in self.shared_params.keys():
            self.adaptive_thresholds[name] = self.initial_threshold

    def update_adaptive_threshold(self):
        # Dynamically adjust threshold for each layer based on gradient magnitude
        for name, p in self.param_history.items():
            mean_grad = torch.mean(torch.abs(p))
            std_grad = torch.std(torch.abs(p))

            # Update layer-specific threshold as mean + k * std (e.g., k=0.5)
            self.adaptive_thresholds[name] = mean_grad + self.k * std_grad

    def apply_updates(self, p, name, group, task_id, sensitive_mask):
        # Step 1: Find protective candidate mask (A = G AND ¬T)
        protective_candidate_mask = self.global_mask[name] & ~self.task_masks[task_id][name]

        # Step 2: Find protective parameters (P = A AND S)
        protective_mask = protective_candidate_mask & sensitive_mask

        # Step 3: Apply protective updates
        if protective_mask.any():
            # Calculate the proximal gradient term: 2 * mu * (theta - theta_0)
            proximal_grad = 2 * self.mu * (p.data - self.initial_params[name])
            # Apply protective update: Use gradient + proximal term
            p.data[protective_mask] -= self.protect_factor * group['lr'] * (p.grad[protective_mask] + proximal_grad[protective_mask])



        p.data[protective_mask] -= self.protect_factor * group['lr'] * p.grad[protective_mask]

        # Step 4: Find task-specific parameters (B = S AND ¬P)
        task_specific_mask = sensitive_mask & ~protective_mask

        # Step 5: Apply normal updates to task-specific parameters
        p.data[task_specific_mask] -= group['lr'] * p.grad[task_specific_mask]

        # Step 6: Update task mask (T = T OR B)
        self.task_masks[task_id][name] = self.task_masks[task_id][name] | task_specific_mask

        # Step 7: Update global mask (G = G OR B)
        self.global_mask[name] = self.global_mask[name] | task_specific_mask

        return task_specific_mask.sum().item(), protective_mask.sum().item()

    def reorder_task_masks(self, sorted_task_ids):
        # Reorder the task masks based on sorted_task_ids
        new_task_masks = {i: self.task_masks[sorted_task_ids[i]] for i in range(len(sorted_task_ids))}
        self.task_masks = new_task_masks

    def calculate_percentages(self, task_id):
        task_related = 0
        protected = 0
        unclaimed = 0

        for name, p in self.shared_params.items():
            # Task-related parameters: those marked in the task-specific mask for the current task
            task_related += self.task_masks[task_id][name].sum().item()

            # Protected parameters: parameters updated by another task (global mask AND not in task mask)
            protected += (~self.task_masks[task_id][name] & self.global_mask[name]).sum().item()

            # Unclaimed parameters: parameters not updated by any task (global mask is zero)
            unclaimed += (~self.global_mask[name]).sum().item()

        # Calculate percentages based on total number of elements
        total_elements = sum(p.numel() for p in self.shared_params.values())
        task_related_percent = (task_related / total_elements) * 100
        protected_percent = (protected / total_elements) * 100
        unclaimed_percent = (unclaimed / total_elements) * 100

        return task_related_percent, protected_percent, unclaimed_percent

    def step(self, task_id, epoch, closure=None):
        # Update layer-wise thresholds after the first epoch
        if epoch > 5:
            self.update_adaptive_threshold()

        total_task_related = 0
        total_protected = 0

        for group in self.param_groups:
            for p, name in zip(group['params'], self.global_mask.keys()):
                if p.grad is None:
                    continue

                # self.param_history[name] += p.grad
                self.param_history[name] = self.beta * self.param_history[name] + (1 - self.beta) * p.grad # EMA
                # Calculate sensitive mask based on accumulated gradients and layer-wise thresholds
                sensitive_mask = torch.abs(self.param_history[name]) > self.adaptive_thresholds[name]


                # Calculate sensitive mask based on layer-wise threshold
                # sensitive_mask = torch.abs(p.grad) > self.adaptive_thresholds[name]

                # Apply both normal and protective updates
                task_related_elements, protective_elements = self.apply_updates(p, name, group, task_id, sensitive_mask)

                total_task_related += task_related_elements
                total_protected += protective_elements


        # if (epoch + 1) % 2 == 0:  # Reset after every 20 epochs
        #    self.param_history = {name: torch.zeros_like(p) for name, p in self.param_history.items()}
 
        super().step(closure)



