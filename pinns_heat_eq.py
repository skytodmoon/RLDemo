import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                self.layers.append(nn.Tanh())
    
    def forward(self, x, t):
        x = torch.cat([x, t], dim=1)
        for layer in self.layers:
            x = layer(x)
        return x

class HeatEquationSolver:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PINN([2, 50, 50, 50, 1]).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.history = []
    
    def loss_fn(self, x_pde, t_pde, x_bc, t_bc, x_ic, t_ic):
        self.model.train()
        
        x_pde.requires_grad_(True)
        t_pde.requires_grad_(True)
        
        u_pde = self.model(x_pde, t_pde)
        u_bc = self.model(x_bc, t_bc)
        u_ic = self.model(x_ic, t_ic)
        
        du_dt = torch.autograd.grad(u_pde, t_pde, grad_outputs=torch.ones_like(u_pde), create_graph=True)[0]
        du_dx = torch.autograd.grad(u_pde, x_pde, grad_outputs=torch.ones_like(u_pde), create_graph=True)[0]
        d2u_dx2 = torch.autograd.grad(du_dx, x_pde, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]
        
        pde_loss = torch.mean((du_dt - self.alpha * d2u_dx2)**2)
        bc_loss = torch.mean(u_bc**2)
        ic_loss = torch.mean((u_ic - torch.sin(np.pi * x_ic))**2)
        
        total_loss = pde_loss + bc_loss + ic_loss
        return total_loss, pde_loss.item(), bc_loss.item(), ic_loss.item()
    
    def train(self, epochs=5000, num_pde=1000, num_bc=200, num_ic=200):
        for epoch in range(epochs):
            x_pde = torch.rand(num_pde, 1).to(self.device)
            t_pde = torch.rand(num_pde, 1).to(self.device)
            
            x_bc = torch.rand(num_bc, 1).to(self.device)
            t_bc = torch.rand(num_bc, 1).to(self.device)
            x_bc_0 = torch.zeros(num_bc//2, 1).to(self.device)
            x_bc_1 = torch.ones(num_bc//2, 1).to(self.device)
            x_bc = torch.cat([x_bc_0, x_bc_1], dim=0)
            
            x_ic = torch.rand(num_ic, 1).to(self.device)
            t_ic = torch.zeros(num_ic, 1).to(self.device)
            
            total_loss, pde_loss, bc_loss, ic_loss = self.loss_fn(x_pde, t_pde, x_bc, t_bc, x_ic, t_ic)
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            if epoch % 100 == 0:
                self.history.append({
                    'epoch': epoch,
                    'total_loss': total_loss.item(),
                    'pde_loss': pde_loss,
                    'bc_loss': bc_loss,
                    'ic_loss': ic_loss
                })
                print(f"Epoch {epoch}: Total Loss={total_loss.item():.6f} "
                      f"PDE={pde_loss:.6f} BC={bc_loss:.6f} IC={ic_loss:.6f}")
    
    def predict(self, x, t):
        self.model.eval()
        x = torch.tensor(x, dtype=torch.float32).reshape(-1, 1).to(self.device)
        t = torch.tensor(t, dtype=torch.float32).reshape(-1, 1).to(self.device)
        with torch.no_grad():
            u = self.model(x, t)
        return u.cpu().numpy().flatten()
    
    def plot_results(self, save_path='pinns_results.png'):
        x = np.linspace(0, 1, 100)
        times = [0.01, 0.1, 0.2, 0.5]
        
        plt.figure(figsize=(10, 6))
        for t in times:
            u_pred = self.predict(x, np.full_like(x, t))
            plt.plot(x, u_pred, label=f't={t}')
        
        plt.xlabel('x')
        plt.ylabel('u(x, t)')
        plt.title('PINNs Solution to Heat Equation')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        print(f"Results saved to {save_path}")
    
    def create_animation(self, save_path='pinns_animation.gif'):
        x = np.linspace(0, 1, 100)
        times = np.linspace(0, 1, 50)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        line, = ax.plot(x, self.predict(x, np.zeros_like(x)), 'b-')
        ax.set_xlabel('x')
        ax.set_ylabel('Temperature')
        ax.set_title('Heat Equation Solution (PINNs)')
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True)
        
        def update(frame):
            t = times[frame]
            u_pred = self.predict(x, np.full_like(x, t))
            line.set_ydata(u_pred)
            ax.set_title(f'Heat Equation Solution (t={t:.2f})')
            return line,
        
        anim = FuncAnimation(fig, update, frames=len(times), interval=100, blit=True)
        anim.save(save_path, writer='pillow')
        plt.close()
        print(f"Animation saved to {save_path}")

if __name__ == '__main__':
    solver = HeatEquationSolver()
    print("Training PINNs for Heat Equation...")
    solver.train(epochs=5000)
    solver.plot_results()
    solver.create_animation()
    print("Training complete!")