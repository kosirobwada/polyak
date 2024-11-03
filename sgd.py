import torch
from torch.optim import Optimizer


class SGD(Optimizer):
    def __init__(self, params, lr, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False):
        if lr < 0.0:
            raise ValueError(f"Learning rate must be non-negative, but got {lr}.")
        if momentum < 0.0:
            raise ValueError(f"Momentum must be non-negative, but got {momentum}.")
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)

    def step(self, closure=None, iteration=0):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if iteration != 0:
            gradient_list = []

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            dampening = group['dampening']
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']

            for param in group['params']:
                if param.grad is None:
                    continue
                gradient = param.grad.data

                if weight_decay != 0:
                    gradient.add_(param.data, alpha=weight_decay)

                if momentum != 0:
                    param_state = self.state[param]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(gradient).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(gradient, alpha=1 - dampening)

                    if nesterov:
                        gradient = gradient.add(buf, alpha=momentum)
                    else:
                        gradient = buf

                if iteration != 0:
                    gradient_list.append(gradient)

                param.data.add_(gradient, alpha=-lr)

        if iteration != 0:
            return gradient_list
        else:
            return loss
