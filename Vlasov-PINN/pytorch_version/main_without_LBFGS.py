import torch
import numpy as np
import os
from net import DNN


# the physics-guided neural network
class PhysicsInformedNN():
    def __init__(self, X, E, f, layers):

        # data
        self.t = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        self.x = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        self.v = torch.tensor(X[:, 2:3], requires_grad=True).float().to(device)
        self.E = torch.tensor(E).float().to(device)
        self.f = torch.tensor(f).float().to(device)

        # settings
        self.lambda_1 = torch.tensor([0.0], requires_grad=True).to(device)
        self.lambda_2 = torch.tensor([0.0], requires_grad=True).to(device)

        self.lambda_1 = torch.nn.Parameter(self.lambda_1)
        self.lambda_2 = torch.nn.Parameter(self.lambda_2)

        # deep neural networks
        self.dnn = DNN(layers).to(device)
        self.dnn.register_parameter('lambda_1', self.lambda_1)
        self.dnn.register_parameter('lambda_2', self.lambda_2)

        self.optimizer = torch.optim.Adam(self.dnn.parameters())

    def net_f(self, t, x, v):
        result = self.dnn(torch.cat([t, x, v], dim=1))
        f = result[:, 0:1]
        E = result[:, 1:2]
        return f, E

    def net_g(self, t, x, v):
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        f, E = self.net_f(t, x, v)

        f_t = torch.autograd.grad(f, t, grad_outputs=torch.ones_like(f), retain_graph=True, create_graph=True)[0]
        f_x = torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), retain_graph=True, create_graph=True)[0]
        f_v = torch.autograd.grad(f, v, grad_outputs=torch.ones_like(f), retain_graph=True, create_graph=True)[0]

        g = f_t + lambda_1 * v * f_x + lambda_2 * E * f_v
        return g

    def train(self, nIter):
        self.dnn.train()
        train_loss_list = []
        for epoch in range(nIter):
            f_pred, E_pred = self.net_f(self.t, self.x, self.v)
            g_pred = self.net_g(self.t, self.x, self.v)
            loss_f = torch.mean((self.f - f_pred) ** 2)
            loss_E = torch.mean((self.E - E_pred) ** 2)
            loss_g = torch.mean((g_pred) ** 2)
            alpha, beta, gamma = (1, 1, 1)
            loss = alpha * loss_g + beta * loss_f + gamma * loss_E

            train_loss_list.append(loss.item())
            with open(os.path.join(save_path, "train_loss.txt"), "a") as f:
                f.write('%.6e' % (loss.item()) + "\n")
            with open(os.path.join(save_path, "loss_f.txt"), "a") as f:
                f.write('%.6e' % (loss_f.item()) + "\n")
            with open(os.path.join(save_path, "loss_E.txt"), "a") as f:
                f.write('%.6e' % (loss_E.item()) + "\n")
            with open(os.path.join(save_path, "loss_g.txt"), "a") as f:
                f.write('%.6e' % (loss_g.item()) + "\n")
            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % 100 == 0:
                print(
                    'Epoch: %d, Loss: %.6e, Lambda_1: %.6f, Lambda_2: %.6f' %
                    (
                        epoch,
                        loss.item(),
                        self.lambda_1.item(),
                        self.lambda_2.item()
                    )
                )
            if (epoch % 500 == 0):
                if loss.item() <= min(train_loss_list):
                    torch.save(self.dnn.state_dict(), os.path.join(save_path, "PINN_" + str(epoch) + ".pth"))
                    print("Model saved!")
            if (epoch % 5000 == 0):
                torch.save(self.dnn.state_dict(), os.path.join(save_path, "PINN_" + str(epoch) + ".pth"))
                print("Model saved!")

    def load_model(self, model_path):
        self.dnn.load_state_dict(torch.load(os.path.join(save_path, model_path)))

np.random.seed(100)

# path
save_path = r"./result"
if not os.path.exists(save_path):
    os.mkdir(save_path)

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

N_u = 3000
layers = [3, 100, 100, 100, 100, 100, 100, 100, 100, 2]

E = np.load(r"./data/E.npy")
f = np.load(r"./data/f.npy")
t_range = (0, 101)
x_range = (100, 150)
v_range = (100, 150)
t = np.linspace(0, 62.5, E.shape[0])[t_range[0]:t_range[1]]
x = np.linspace(0, 10, 257)[x_range[0]:x_range[1]]
v = np.linspace(-5, 5, 257)[v_range[0]:v_range[1]]
E = np.swapaxes(E, 1, 2)[t_range[0]:t_range[1], x_range[0]:x_range[1], v_range[0]:v_range[1]]
f = np.swapaxes(f, 1, 2)[t_range[0]:t_range[1], x_range[0]:x_range[1], v_range[0]:v_range[1]]

T, X, V = np.meshgrid(t, x, v)
T = np.swapaxes(T, 0, 1)
X = np.swapaxes(X, 0, 1)
V = np.swapaxes(V, 0, 1)

X_star = np.hstack((T.flatten()[:, None], X.flatten()[:, None], V.flatten()[:, None]))  # t,x,v
f_star = f.flatten()[:, None]

# create training set
idx = np.random.choice(X_star.shape[0], N_u, replace=False)
X_u_train = X_star[idx, :]
idx_on_tx = np.unravel_index(idx, f.shape)  # return index in tx space
E_train = E[idx_on_tx[0], idx_on_tx[1]]
f_train = f_star[idx, :]

step = 10000

# training
model = PhysicsInformedNN(X_u_train, E_train, f_train, layers)
model.train(step)