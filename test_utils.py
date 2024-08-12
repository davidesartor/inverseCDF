import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy


def run_test(models, test_case, device='cuda', max_steps=10000, lr=0.01, early_stop_thr=0.0, copy=True):
    x, y = get_data(test_case)
    if copy:
        models = {name: deepcopy(model) for name, model in models.items()}
    models, losses = train(models, x, y, device, max_steps, lr, early_stop_thr)
    models = {name: model.cpu() for name, model in models.items()}
    plot_results(models, x, y, losses)
    return models, x, y, losses


def get_data(test_case):
    if test_case == 'decreasing':
        x = torch.linspace(-2, 2, 1024).unsqueeze(-1)
        y = -torch.sign(x)*x**2
    elif test_case == 'easy1d':
        x = torch.linspace(-2, 2, 1024).unsqueeze(-1)
        y = torch.sign(x)*x**2
    elif test_case == 'hard1d':
        x = torch.linspace(-2, 2, 1024).unsqueeze(-1)
        y = torch.where(x>0, torch.sin(6.28*x)/6.28+x, 0.5+x.floor()) 
    elif test_case == 'easy2d':
        eps=1e-1
        x = 6*(torch.rand(1024, 2)-0.5)
        t1 = (x[:, 0] > eps) * (x[:, 1] > eps)
        t2 = (x[:, 0] > -eps) * (x[:, 1] > -eps)
        y = ((t1.float()+t2.float())/2).unsqueeze(-1)
    else:
        raise ValueError(f'Unknown test case: {test_case}')
    return x, y

def train(models, x, y, device='cuda', max_steps=1000, lr=0.01, early_stop_thr=0.0):
    losses = {model: [] for model in models.keys()}
    for name, model in models.items():
        model = model.to(device)
        x, y = x.to(device), y.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for i in (pbar:=tqdm(range(max_steps), desc=name)):
            loss = ((model(x) - y)**2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses[name].append(loss.item()**0.5)
            rmse = sum(losses[name][-10:])/10
            if i%10==0:
                pbar.set_postfix({'rmse': rmse})
            if rmse < early_stop_thr:
                break            
    return models, losses


def plot_losses(losses, legend=True):
    for name, loss in losses.items():
        plt.semilogy(loss, label=name)
    if legend: plt.legend(loc='upper right')
    plt.grid()


def plot_predictions1d(models, x, y, legend=True):
    plt.plot(x, y, "k", label='target')
    for name, model in models.items():
        pred = model(x).detach()
        rmse = ((pred - y)**2).mean().item()**0.5
        plt.plot(x, pred, label=f'{name} (rmse={rmse:.2f})')
    plt.grid()
    if legend: plt.legend(loc='upper left')


def plot_predictions2d(model, x, y, title=''):
    plt.plot(x[:, 0], x[:, 1], 'k.', alpha=0.1)
    xmin, xmax = x[:, 0].min(), x[:, 0].max()
    ymin, ymax = x[:, 1].min(), x[:, 1].max()
    X, Y = torch.meshgrid(
        torch.linspace(xmin, xmax, 100), 
        torch.linspace(ymin, ymax, 100),
        indexing='ij'
    ) 
    x = torch.stack((X, Y), dim=-1).reshape(-1, 2)
    plt.imshow(
        model(x).detach().reshape(100, 100), 
        extent=(xmin, xmax, ymin, ymax),
        origin='lower', cmap='coolwarm'
    )
    plt.title(title)
    plt.colorbar()
    

def plot_results(models, x, y, losses):
    if x.shape[-1] == 1:
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plot_predictions1d(models, x, y, legend=False)
        plt.subplot(122)
        plot_losses(losses, legend=True)
        plt.show()
    if x.shape[-1] == 2:
        k = int(torch.ceil(torch.tensor(len(models)/2)))
        plt.figure(figsize=(4*k, 3*k))
        for i, (name, model) in enumerate(models.items()):
            plt.subplot(2, k, i+1)
            plot_predictions2d(model, x, y, title=name)
        plt.figure(figsize=(5, 5))
        plot_losses(losses, legend=True)
        plt.show()