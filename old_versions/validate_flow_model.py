import torch
import torch.utils.data as data
import torchdiffeq
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms as transforms

from conditional_mnist_tests import load_mnist_data, get_device
from tree_flow_matching import SigmaConditionedUNet
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
import numpy as np
import os
from evaluation import visualize_class_samples



def generate_noise_levels(num_levels=10):
    """Generate noise levels with more density near 0."""
    # Generate exponentially spaced values between 0 and 1
    x = np.exp(np.linspace(np.log(1e-4), np.log(1.0), num_levels))
    # Reverse the order so we go from 1.0 down to small values
    x = np.flip(x)
    return x.tolist()

class TreeFlowModel:
    def __init__(self, device=None):
        self.device = device if device is not None else get_device()
        self.save_dir = "saved_tree_model"
        self.save_path = os.path.join(self.save_dir, "tree_flow_model.pt")
        
        # Initialize model
        self.model = SigmaConditionedUNet(
            dim=(1, 28, 28),
            num_channels=32,
            num_res_blocks=1,
            num_classes=10,
            class_cond=True
        ).to(self.device)
        
        # Generate noise levels
        self.noise_levels = generate_noise_levels(5)
        
    def train(self, n_epochs=3, batch_size=4,max_batches=50):
        # Check if saved model exists
        if os.path.exists(self.save_path):
            print("Loading saved model...")
            self.model.load_state_dict(torch.load(self.save_path, weights_only=True))
            return self
            
        print("Training new model...")
        print("Noise levels:", [f"{x:.5f}" for x in self.noise_levels])
        
        # Load data
        train_loader = load_mnist_data(batch_size=batch_size, max_batches=max_batches)
        
        optimizer = torch.optim.Adam(self.model.parameters())
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
        
        # Training loop
        losses = {i: [] for i in range(len(self.noise_levels)-1)}
        
        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch + 1}/{n_epochs}")
            epoch_losses = {i: [] for i in range(len(self.noise_levels)-1)}
            
            pbar = tqdm(train_loader)
            for batch_idx, data in enumerate(pbar):
                x1, y = data[0].to(self.device), data[1].to(self.device)
                
                if torch.rand(1).item() < 0.75:
                    transition_idx = 0
                else:
                    transition_idx = 1 + (batch_idx % (len(self.noise_levels)-2))
                
                current_sigma = self.noise_levels[transition_idx]
                next_sigma = self.noise_levels[transition_idx + 1]
                
                # Generate start and end points
                if current_sigma == 1.0:
                    x0 = torch.randn_like(x1)
                else:
                    x0 = x1 + torch.randn_like(x1) * current_sigma
                    
                x_target = x1 + torch.randn_like(x1) * next_sigma
                
                # Train flow matching
                optimizer.zero_grad()
                t, xt, ut = FM.sample_location_and_conditional_flow(x0, x_target)
                
                sigma = torch.full((x1.shape[0],), current_sigma, device=self.device)
                vt = self.model(t, xt, y, sigma=sigma)
                
                loss = torch.mean((vt - ut) ** 2)
                loss.backward()
                optimizer.step()
                
                epoch_losses[transition_idx].append(loss.item())
                
                pbar.set_postfix({
                    'transition': f'{current_sigma:.5f}->{next_sigma:.5f}',
                    'loss': f'{loss.item():.4f}'
                })
            
            # Log average losses
            for idx in epoch_losses:
                avg_loss = (sum(epoch_losses[idx]) / len(epoch_losses[idx])) if len(epoch_losses[idx]) > 0 else 0
                losses[idx].append(avg_loss)
                print(f"Transition {self.noise_levels[idx]:.5f}->{self.noise_levels[idx+1]:.5f} "
                      f"average loss: {avg_loss:.4f}")
        
        # Save the model
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), self.save_path)
        print(f"\nModel saved to {self.save_path}")
        
        self.plot_losses(losses)
        return self
    
    def sample(self, num_samples=16, class_labels=None):
        self.model.eval()
        with torch.no_grad():
            if class_labels is None:
                class_labels = torch.randint(0, 10, (num_samples,), device=self.device)
            
            current = torch.randn(num_samples, 1, 28, 28, device=self.device)
            
            for i in range(len(self.noise_levels)-1):
                current_sigma = self.noise_levels[i]
                sigma = torch.full((num_samples,), current_sigma, device=self.device)
                
                traj = torchdiffeq.odeint(
                    lambda t, x: self.model(t, x, class_labels, sigma=sigma),
                    current,
                    torch.linspace(0, 1, 2, device=self.device),
                    atol=1e-4, rtol=1e-4
                )
                current = traj[-1]
            
            return current
    
    def plot_losses(self, losses):
        plt.figure(figsize=(10, 5))
        for idx in losses:
            plt.plot(losses[idx], 
                    label=f'{self.noise_levels[idx]:.5f}->{self.noise_levels[idx+1]:.5f}')
        plt.title('Training Loss by Transition')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def generate_samples_by_class(self, samples_per_class=16):
        """Generate samples for each class."""
        all_samples = []
        all_labels = []
        
        for digit in range(10):
            # Generate samples for this digit
            class_labels = torch.full((samples_per_class,), digit, device=self.device)
            samples = self.sample(num_samples=samples_per_class, class_labels=class_labels)
            
            all_samples.append(samples)
            all_labels.append(class_labels)
        
        # Combine all samples and labels
        return torch.cat(all_samples), torch.cat(all_labels)

if __name__ == "__main__":
    model = TreeFlowModel()
    model.train(n_epochs=20, batch_size=64, max_batches=200)
    
    # Generate and visualize samples
    print("\nGenerating samples...")
    class_labels = torch.arange(10, device=model.device).repeat(4)
    samples = model.sample(num_samples=40, class_labels=class_labels)

    plt.figure(figsize=(15, 6))
    grid = make_grid(samples, nrow=10, normalize=True, padding=2)
    plt.imshow(grid.cpu().permute(1, 2, 0))
    plt.title('Generated Samples')
    plt.axis('off')
    plt.show()

    samples, labels = model.generate_samples_by_class(samples_per_class=16)
    visualize_class_samples(samples, labels)