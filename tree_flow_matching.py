import torch
import torch.nn as nn
from torch.nn import MSELoss
import torchdiffeq
from tqdm import tqdm
from torchcfm.models.unet import UNetModel
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from torchcfm.models.unet.nn import timestep_embedding

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
    
    def train_epoch(self, train_loader):
        """Train both flow and value models for one epoch."""
        # Initialize datasets for each depth
        depth_datasets = {0: [(x, y) for x, y in train_loader]}
        
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
        
        for i, (x1, y) in enumerate(pbar):
            x1, y = x1.to(self.device), y.to(self.device)
            self.flow_optimizer.zero_grad()
            
            # At depth 0, start from random noise like standard CFM
            # At deeper levels, start from x1 plus scaled noise
            if depth == 0:
                x0 = torch.randn_like(x1)
                sigma = torch.ones(x1.shape[0], device=self.device)  # Full noise
            else:
                sigma_estimate = 1.0 / (depth + 1)
                x0 = x1 + torch.randn_like(x1) * sigma_estimate
                sigma = torch.full((x1.shape[0],), sigma_estimate, device=self.device)
            
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
            for x1, y in ode_pbar:
                x1, y = x1.to(self.device), y.to(self.device)
                x0 = torch.randn_like(x1)
                
                # Generate sample using ODE
                traj = torchdiffeq.odeint(
                    lambda t, x: self.flow_model(t, x, y),
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

        # Now train value model using ground truth + sampled noise
        value_pbar = tqdm(dataset, desc=f"Training value model (depth {depth})")
        for i, (x1, y) in enumerate(value_pbar):
            x1, y = x1.to(self.device), y.to(self.device)
            
            # Sample noise level from fitted Gaussian
            sampled_noise = torch.normal(noise_mean, noise_std)
            breakpoint()
            
            # Create noisy sample by adding appropriate noise to ground truth
            noise = torch.randn_like(x1)
            noisy_sample = x1 + noise * sampled_noise
            
            # Train value model
            self.value_optimizer.zero_grad()
            noise_estimate = self.value_model(
                torch.ones(1, device=self.device),
                noisy_sample,
                y
            )
            value_loss = self.value_criterion(noise_estimate, sampled_noise)
            value_loss.backward()
            self.value_optimizer.step()
            value_losses.append(value_loss.item())
            
            value_pbar.set_postfix({'loss': f'{value_loss.item():.4f}'})
            
            # Add to next depth's dataset
            next_depth_samples.append((noisy_sample.detach(), y))
        
        return next_depth_samples, value_losses
  
    def sample(self, num_samples, class_labels, 
            num_branches=5, num_select=2, num_steps=3):
        """
        Generate samples using tree-based exploration guided by value model.
        
        Args:
            num_samples: Number of final samples to generate
            class_labels: Class conditioning for each sample
            num_branches: Number of branches to create at each node
            num_select: Number of best branches to keep at each step
            num_steps: Number of refinement steps
        """
        self.flow_model.eval()
        self.value_model.eval()
        
        with torch.no_grad():
            # Initialize samples from noise
            current_samples = torch.randn(
                num_samples, 1, 28, 28, 
                device=self.device
            )
            
            for step in range(num_steps):
                next_samples = []
                next_scores = []
                
                # For each current sample
                for i in range(len(current_samples)):
                    x0 = current_samples[i]
                    y = class_labels[i]
                    
                    branch_samples = []
                    branch_scores = []
                    
                    # Generate branches by adding small perturbations
                    for _ in range(num_branches):
                        # Add decreasing noise for each step
                        noise_scale = 0.1 * (num_steps - step) / num_steps
                        perturbed_x0 = x0 + torch.randn_like(x0) * noise_scale
                        
                        # Generate sample using flow model
                        traj = torchdiffeq.odeint(
                            lambda t, x: self.flow_model(t, x, y.unsqueeze(0)),
                            perturbed_x0.unsqueeze(0),
                            torch.linspace(0, 1, 2, device=self.device),
                            atol=1e-4, rtol=1e-4
                        )
                        generated = traj[-1][0]  # Remove batch dimension
                        
                        # Get value estimate
                        noise_estimate = self.value_model(
                            torch.ones(1, device=self.device),
                            generated.unsqueeze(0),
                            y.unsqueeze(0)
                        )
                        
                        branch_samples.append(generated)
                        branch_scores.append(noise_estimate.item())
                    
                    # Select best branches
                    branch_samples = torch.stack(branch_samples)
                    branch_scores = torch.tensor(branch_scores)
                    best_indices = torch.topk(
                        -branch_scores,  # Negative because we want lowest noise
                        k=min(num_select, num_branches)
                    ).indices
                    
                    next_samples.extend(branch_samples[best_indices])
                    next_scores.extend(branch_scores[best_indices])
                
                # Prepare for next step: select overall best samples
                if step < num_steps - 1:
                    next_samples = torch.stack(next_samples)
                    next_scores = torch.tensor(next_scores)
                    best_indices = torch.topk(
                        -next_scores,
                        k=num_samples
                    ).indices
                    current_samples = next_samples[best_indices]
                else:
                    # For final step, return all samples
                    current_samples = torch.stack(next_samples)
            
            return current_samples

    # def sample(self, num_samples, class_labels):
    #     """
    #     Simple sampling using just one forward pass through flow model.
        
    #     Args:
    #         num_samples: Number of samples to generate
    #         class_labels: Class conditioning for each sample
    #     """
    #     self.flow_model.eval()
        
    #     with torch.no_grad():
    #         # Initialize samples from noise
    #         x0 = torch.randn(num_samples, 1, 28, 28, device=self.device)
            
    #         # Generate samples using flow model
    #         traj = torchdiffeq.odeint(
    #             lambda t, x: self.flow_model(t, x, class_labels),
    #             x0,
    #             torch.linspace(0, 1, 2, device=self.device),
    #             atol=1e-4, rtol=1e-4
    #         )
    #         generated_samples = traj[-1]
            
    #         return generated_samples

class ValueModel(UNetModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_layer = torch.nn.Conv2d(self.out_channels, 1, 1)
        
    def forward(self, t, x, y):
        features = super().forward(t, x, y)
        return self.final_layer(features).mean()
    

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
