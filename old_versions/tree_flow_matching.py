from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.nn import MSELoss
import torchdiffeq
from tqdm import tqdm
from torchcfm.models.unet import UNetModel
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher, TargetConditionalFlowMatcher
from torchcfm.models.unet.nn import timestep_embedding
from torchvision.utils import make_grid

class TreeFlowMatching:
    def __init__(
        self,
        num_channels=32,
        num_res_blocks=1,
        num_classes=10,
        num_branches=5,
        max_depth=3,
        samples_per_depth=100,
        sigma=0.0,
        device=None
    ):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize policy (flow) and value models
        self.flow_model = SigmaConditionedUNet(
            dim=(1, 28, 28),  # Changed to match standard_cfm format
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
            num_classes=num_classes,
            class_cond=True
        ).to(self.device)
        
        self.value_model = ValueModel(
            dim=(1, 28, 28),  # Changed to match standard_cfm format
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
            num_classes=num_classes,
            class_cond=True
        ).to(self.device)
        
        # Training parameters
        self.num_branches = num_branches
        self.max_depth = max_depth
        self.samples_per_depth = samples_per_depth
        
        # Initialize optimizers and loss functions
        self.flow_optimizer = torch.optim.Adam(self.flow_model.parameters())
        self.value_optimizer = torch.optim.Adam(self.value_model.parameters())
        self.value_criterion = MSELoss()
        self.FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
        
        # Training metrics
        self.flow_losses = []
        self.value_losses = []

    def _add_noise(self, x1, sigma):
        """Add scaled Gaussian noise to samples.
        
        Args:
            x1 (torch.Tensor): Input samples to add noise to
            sigma (torch.Tensor): Noise scale for each sample in batch
                (represents amount of noise to add)
            
        Returns:
            torch.Tensor: Noisy samples with approximately sigma amount of noise
        """
        noise = torch.randn_like(x1)
        return x1 + noise * sigma.view(-1, 1, 1, 1)

    def train_epoch(self, train_loader):
        """Train both flow and value models for one epoch."""
        # Initialize datasets for depth 0 with sigma=1.0 for full noise
        depth_datasets = {0: [(x, y, torch.ones(x.shape[0], device=self.device)) 
                            for x, y in train_loader]}
        
        for depth in range(self.max_depth):
            print(f"\nTraining at depth {depth}")
            
            # Train flow model at current depth
            flow_losses_depth = self._train_flow_model(depth_datasets[depth], depth)
            self.flow_losses.extend(flow_losses_depth)
            
            # Generate samples and train value model for next depth
            if depth < self.max_depth - 1:
                next_depth_samples, value_losses = self._generate_samples_and_train_value(
                    depth_datasets[depth],
                    depth
                )
                depth_datasets[depth + 1] = next_depth_samples
                self.value_losses.extend(value_losses)
    
    def _train_flow_model(self, dataset, depth):
        """Train flow model at current depth."""
        self.flow_model.train()
        losses = []
        pbar = tqdm(dataset, desc=f"Training flow model (depth {depth})")
        
        for i, batch in enumerate(pbar):
            x1, y, sigma = batch
            x1, y, sigma = x1.to(self.device), y.to(self.device), sigma.to(self.device)
            
            if depth == 0:
                x0 = torch.randn_like(x1)  # For depth 0, we start from pure noise
            else:
                x0 = self._add_noise(x1, sigma)
            
            self.flow_optimizer.zero_grad()
            t, xt, ut = self.FM.sample_location_and_conditional_flow(x0, x1)
            vt = self.flow_model(t, xt, y, sigma=sigma)
            flow_loss = torch.mean((vt - ut) ** 2)
            
            flow_loss.backward()
            self.flow_optimizer.step()
            losses.append(flow_loss.item())
            
            pbar.set_postfix({'loss': f'{flow_loss.item():.4f}'})
        
        return losses

    def _generate_samples_and_train_value(self, dataset, depth, num_ode_samples=1):
        """Generate samples using flow model and train value model efficiently."""
        self.flow_model.eval()
        self.value_model.train()
        next_depth_samples = []
        value_losses = []
        
        # First, generate ODE samples to estimate noise distribution
        noise_values = []
        ode_pbar = tqdm(dataset[:num_ode_samples], desc=f"Generating ODE samples (depth {depth})")
        with torch.no_grad():
            for x1, y, sigma in ode_pbar:
                x1, y, sigma = x1.to(self.device), y.to(self.device), sigma.to(self.device)
                x0 = self._add_noise(x1, sigma)
                
                # Generate sample using ODE with sigma conditioning
                traj = torchdiffeq.odeint(
                    lambda t, x: self.flow_model(t, x, y, sigma=sigma),
                    x0,
                    torch.linspace(0, 1, 2, device=self.device),
                    atol=1e-4, rtol=1e-4
                )
                generated = traj[-1]
                
                # Calculate noise level for each sample in batch
                diff_flat = (generated - x1).view(x1.shape[0], -1)
                x1_flat = x1.view(x1.shape[0], -1)
                
                noise_levels = torch.norm(diff_flat, dim=1) / torch.norm(x1_flat, dim=1)
                noise_values.extend(noise_levels.tolist())

        # Fit Gaussian to all noise values across batches
        noise_mean = torch.tensor(noise_values).mean()
        noise_std = torch.tensor(noise_values).std()

        print(f"\nDepth {depth} ODE-generated Noise Statistics:")
        print(f"Mean relative noise: {noise_mean:.4f}")
        print(f"Std relative noise: {noise_std:.4f}")

        # Now train value model using ground truth + sampled noise
        # Also collect statistics about sampled noise for validation
        sampled_noise_values = []
        value_pbar = tqdm(dataset, desc=f"Training value model (depth {depth})")
        for x1, y, _ in value_pbar:
            x1, y = x1.to(self.device), y.to(self.device)
            
            # Sample noise level from fitted Gaussian
            sampled_noise = torch.normal(
                noise_mean.expand(x1.shape[0]),
                noise_std.expand(x1.shape[0])
            )
            
            # Create noisy sample
            noisy_sample = self._add_noise(x1, sampled_noise)
            
            # Calculate actual noise level for validation
            diff_flat = (noisy_sample - x1).view(x1.shape[0], -1)
            x1_flat = x1.view(x1.shape[0], -1)
            actual_noise_levels = torch.norm(diff_flat, dim=1) / torch.norm(x1_flat, dim=1)
            sampled_noise_values.extend(actual_noise_levels.tolist())
            
            # Train value model
            self.value_optimizer.zero_grad()
            noise_estimate = self.value_model(
                torch.ones(x1.shape[0], device=self.device),
                noisy_sample,
                y
            )
            value_loss = self.value_criterion(noise_estimate, sampled_noise)
            value_loss.backward()
            self.value_optimizer.step()
            value_losses.append(value_loss.item())
            
            value_pbar.set_postfix({'loss': f'{value_loss.item():.4f}'})
            
            # Add to next depth's dataset with true noise levels
            next_depth_samples.append((noisy_sample.detach(), y, sampled_noise.detach()))
        
        # Print statistics about the sampled noise
        sampled_noise_mean = torch.tensor(sampled_noise_values).mean()
        sampled_noise_std = torch.tensor(sampled_noise_values).std()
        print(f"\nDepth {depth} Sampled Noise Statistics:")
        print(f"Mean relative noise: {sampled_noise_mean:.4f}")
        print(f"Std relative noise: {sampled_noise_std:.4f}")
        
        return next_depth_samples, value_losses
  
    def sample(self, num_samples, class_labels, 
            num_branches=5, num_select=2, num_steps=None):
        """Generate samples using tree-based exploration guided by value model."""
        if num_steps is None or num_steps > self.max_depth:
            num_steps = self.max_depth
        self.flow_model.eval()
        self.value_model.eval()
        
        with torch.no_grad():
            current_samples = torch.randn(
                num_samples, 1, 28, 28, 
                device=self.device
            )
            current_sigmas = torch.ones(num_samples, device=self.device)
            
            # Get initial noise estimates
            initial_noise_estimates = self.value_model(
                torch.ones(num_samples, device=self.device),
                current_samples,
                class_labels
            )
            
            # Visualize initial noise
            plt.figure(figsize=(10, 4))
            grid = make_grid(current_samples[:8], nrow=8, normalize=True, padding=2)
            plt.imshow(grid.cpu().permute(1, 2, 0))
            plt.title(f'Initial Noise Samples\nEstimated noise levels: ' + 
                    ', '.join([f'{x:.3f}' for x in initial_noise_estimates[:8].cpu()]))
            plt.axis('off')
            plt.show()
            
            pbar = tqdm(range(num_steps), desc="Generating samples")
            for step in pbar:
                # Create all perturbed samples at once
                noise_scale = 0.1 * (num_steps - step) / num_steps
                # Repeat each sample num_branches times
                x0_expanded = current_samples.repeat_interleave(num_branches, dim=0)
                perturbations = torch.randn_like(x0_expanded) * noise_scale
                perturbed_x0 = x0_expanded + perturbations
                
                # Repeat class labels and sigmas for each branch
                y_expanded = class_labels.repeat_interleave(num_branches)
                sigma_expanded = current_sigmas.repeat_interleave(num_branches)
                
                # Generate all samples in one batch
                traj = torchdiffeq.odeint(
                    lambda t, x: self.flow_model(
                        t, x, y_expanded, 
                        sigma=sigma_expanded
                    ),
                    perturbed_x0,
                    torch.linspace(0, 1, 2, device=self.device),
                    atol=1e-4, rtol=1e-4
                )
                generated = traj[-1]
                
                # Get all noise estimates in one batch
                noise_estimates = self.value_model(
                    torch.ones(len(generated), device=self.device),
                    generated,
                    y_expanded
                )
                
                # Reshape to (num_samples, num_branches, ...)
                generated = generated.view(num_samples, num_branches, *generated.shape[1:])
                noise_estimates = noise_estimates.view(num_samples, num_branches)
                
                # Select best branches for each sample
                best_branch_indices = torch.topk(-noise_estimates, k=num_select, dim=1)
                
                # Gather best samples and their scores
                next_samples = []
                next_sigmas = []
                for i in range(num_samples):
                    sample_branches = generated[i]
                    best_indices = best_branch_indices.indices[i]
                    next_samples.extend([sample_branches[j] for j in best_indices])
                    next_sigmas.extend([noise_estimates[i, j] for j in best_indices])
                
                # Stack and select final samples for next iteration
                next_samples = torch.stack(next_samples)
                next_sigmas = torch.stack(next_sigmas)
                
                if step < num_steps - 1:
                    best_indices = torch.topk(-next_sigmas, k=num_samples).indices
                    current_samples = next_samples[best_indices]
                    current_sigmas = next_sigmas[best_indices]
                    pbar.set_postfix({'avg_sigma': f'{current_sigmas.mean().item():.4f}'})
                else:
                    current_samples = next_samples
                
                # Get noise estimates for displayed samples
                display_noise_estimates = self.value_model(
                    torch.ones(len(current_samples), device=self.device),
                    current_samples,
                    class_labels.repeat(num_select) if step == num_steps - 1 else class_labels
                )
                
                # Visualize current state
                plt.figure(figsize=(10, 4))
                grid = make_grid(current_samples[:8], nrow=8, normalize=True, padding=2)
                plt.imshow(grid.cpu().permute(1, 2, 0))
                plt.title(f'After Step {step+1}, Avg Sigma: {current_sigmas.mean().item():.4f}\n' +
                        'Estimated noise levels: ' + 
                        ', '.join([f'{x:.3f}' for x in display_noise_estimates[:8].cpu()]))
                plt.axis('off')
                plt.show()
            
            return current_samples

class ValueModel(UNetModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_layer = torch.nn.Conv2d(self.out_channels, 1, 1)
        
    def forward(self, t, x, y):
        features = super().forward(t, x, y)
        return self.final_layer(features).mean(dim=[1, 2, 3])
    

class SigmaConditionedUNet(UNetModel):
    """UNet model that takes sigma estimation as additional conditioning."""
    def __init__(self, dim, num_channels, num_res_blocks, num_classes=None, class_cond=True):
        super().__init__(
            dim=dim,
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
            class_cond=True,
            num_classes=num_classes if class_cond else None
        )
        
        # Add embedding for sigma conditioning
        time_embed_dim = self.model_channels * 4
        self.sigma_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        # self.num_classes = num_classes

    def forward(self, t, x, y=None, sigma=None):
        """
        Forward pass with sigma conditioning.
        Args:
            t: time step
            x: input tensor
            y: class labels
            sigma: estimated remaining noise (scalar between 0 and 1)
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        
        # Process timesteps
        while t.dim() > 1:
            t = t[:, 0]
        if t.dim() == 0:
            t = t.repeat(x.shape[0])

        # Get embeddings
        time_emb = self.time_embed(timestep_embedding(t, self.model_channels))
        
        if sigma is not None:
            # Process sigma conditioning
            if sigma.dim() == 0:
                sigma = sigma.repeat(x.shape[0])
            sigma_emb = self.sigma_embed(sigma.view(-1, 1))
            # Combine time and sigma embeddings
            emb = time_emb + sigma_emb
        else:
            emb = time_emb

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            # Add class embedding to combined embedding
            emb = emb + self.label_emb(y)

        # Process through UNet
        h = x.type(self.dtype)
        hs = []
        
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            
        h = self.middle_block(h, emb)
        
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
            
        h = h.type(x.dtype)
        return self.out(h)
