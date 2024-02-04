from torch.optim.optimizer import Optimizer
import torch
import math

class AdamW(Optimizer):
    """ Implements Adam algorithm with weight decay fix.
    Parameters:
        lr (float): learning rate. Default 1e-3.
        betas (tuple of 2 floats): Adams beta parameters (b1, b2). Default: (0.9, 0.98)
        eps (float): Adams epsilon. Default: 1e-6
        weight_decay (float): Weight decay. Default: 0.0
        correct_bias (bool): can be set to False to avoid correcting bias in Adam (e.g. like in Bert TF repository). Default True.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.0, correct_bias=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)


    def reset_state(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            i = 0
            j = 0
            for p in group["params"]:
                if p.grad is None:
                    continue
                i += 1
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if 'correct_bias' in group and group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss

class SGDr(Optimizer):
    def __init__(self, params, lr, weight_decay, rank=4):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.rank = rank
    def step(self):
        for group in self.param_groups:
            for p1, p2 in list(zip(group["params"],group["params"][1:]))[::2]:
                dim_1 = p2.data.shape[0]//2
                grad1_0 = p1.grad.data[0:self.rank,:]
                grad1_1 = p1.grad.data[self.rank:,:]
                scale1_0 = p2.data[0:dim_1,:]
                scale1_1 = p2.data[dim_1:,:]
                try:
                    grad1_0_scaled = torch.inverse(scale1_0.T@scale1_0)@grad1_0
                except:
                    grad1_0_scaled = grad1_0
                try:
                    grad1_1_scaled = torch.inverse(scale1_1.T@scale1_1)@grad1_1
                except:
                    grad1_1_scaled = grad1_1
                grad1_scaled = torch.cat([grad1_0_scaled, grad1_1_scaled])

                grad2_0 = p2.grad.data[0:dim_1,:]
                grad2_1 = p2.grad.data[dim_1:,:]
                scale2_0 = p1.data[0:self.rank,:]
                scale2_1 = p1.data[self.rank:,:]
                try:
                    grad2_0_scaled = grad2_0@torch.inverse(scale2_0@scale2_0.T)
                except:
                    grad2_0_scaled = grad2_0
                try:
                    grad2_1_scaled = grad2_1@torch.inverse(scale2_1@scale2_1.T)
                except:
                    grad2_1_scaled = grad2_1
                grad2_scaled = torch.cat([grad2_0_scaled, grad2_1_scaled])

                if group["weight_decay"] > 0.0:
                    p1.data.add_(p1.data, alpha=-group["lr"] * group["weight_decay"])
                    p2.data.add_(p2.data, alpha=-group["lr"] * group["weight_decay"])

                p1.data.add_(grad1_scaled, alpha=-group['lr'])
                p2.data.add_(grad2_scaled, alpha=-group['lr'])


class AdamWr(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.0, correct_bias=True, rank=2):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)
        self.rank = rank
    def reset_state(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p1, p2 in list(zip(group["params"],group["params"][1:]))[::2]:
                state = self.state[p1]
                dim_1 = p2.data.shape[0]//2
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p1.data)
                    state["exp_avg_sq"] = torch.zeros_like(p1.data)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1
                grad1_scaled0 = p1.grad.data[0:self.rank,:]
                c0 = p2.data[0:dim_1,:]
                try:
                    c0_ = torch.inverse(c0.T@c0)
                except:
                    c0_ = torch.eye((c0.T@c0).shape[0]).to(c0.device)
                grad1_scaled1 = p1.grad.data[self.rank:,:]
                c1 = p2.data[dim_1:,:]
                try:
                    c1_ = torch.inverse(c1.T@c1)
                except:
                    c1_ = torch.eye((c1.T@c1).shape[0]).to(c1.device)
                grad1_scaled0 = c0_@grad1_scaled0
                grad1_scaled1 = c1_@grad1_scaled1
                grad1_scaled = torch.cat([grad1_scaled0, grad1_scaled1])
                assert grad1_scaled.shape == p1.grad.data.shape


                exp_avg.mul_(beta1).add_(grad1_scaled, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad1_scaled, grad1_scaled, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if 'correct_bias' in group and group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                c0 = p1.data[0:self.rank,:]
                c1 = p1.data[self.rank:,:]
                if group["weight_decay"] > 0.0:
                    p1.data.add_(p1.data, alpha=-group["lr"] * group["weight_decay"])
                p1.data.addcdiv_(-step_size, exp_avg, denom)
                # if group["weight_decay"] > 0.0:
                #     p1.data.add_(p1.data, alpha=-group["lr"] * group["weight_decay"])

                
                state = self.state[p2]
                
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p2.data)
                    state["exp_avg_sq"] = torch.zeros_like(p2.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                grad2_scaled0 = p2.grad.data[0:dim_1,:]
                try:
                    c0_ = torch.inverse(c0@c0.T)
                except:
                    c0_ = torch.eye((c0@c0.T).shape[0]).to(c0.device)
                grad2_scaled1 = p2.grad.data[dim_1:,:]
                try:
                    c1_ = torch.inverse(c1@c1.T)
                except:
                    c1_ = torch.eye((c1@c1.T).shape[0]).to(c1.device)
                grad2_scaled0 = grad2_scaled0@c0_
                grad2_scaled1 = grad2_scaled1@c1_
                grad2_scaled = torch.cat([grad2_scaled0,grad2_scaled1])
                assert grad2_scaled.shape == p2.grad.data.shape
                
                exp_avg.mul_(beta1).add_(grad2_scaled, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad2_scaled, grad2_scaled, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if 'correct_bias' in group and group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                if group["weight_decay"] > 0.0:
                    p2.data.add_(p2.data, alpha=-group["lr"] * group["weight_decay"])
                p2.data.addcdiv_(-step_size, exp_avg, denom)

        return loss