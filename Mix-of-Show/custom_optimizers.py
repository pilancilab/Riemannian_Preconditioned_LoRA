from torch.optim.optimizer import Optimizer, required
import torch
import math
import copy
import time

class AdamWv(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.0, correct_bias=False):
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
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if 'correct_bias' in group and group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])
                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss

class SGDr(Optimizer):
    def __init__(self, params, lr, weight_decay, betas=(0.9, 0.98), eps=1e-6, correct_bias=True):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)
    def step(self, closure=None):
        for group in self.param_groups:
            for p1, p2 in list(zip(group["params"],group["params"][1:]))[::2]:
                grad1 = p1.grad.data
                scale1 = p2.data
                try:
                    grad1_scaled = torch.inverse(scale1.T@scale1)@grad1
                except:
                    grad1_scaled = grad1
                
                grad2 = p2.grad.data
                scale2 = p1.data
                try:
                    grad2_scaled = grad2@torch.inverse(scale2@scale2.T)
                except:
                    grad2_scaled = grad2
                
                if group["weight_decay"] > 0.0:
                    p1.data.add_(p1.data, alpha=-group["lr"] * group["weight_decay"])
                    p2.data.add_(p2.data, alpha=-group["lr"] * group["weight_decay"])

                p1.data.add_(grad1_scaled, alpha=-group['lr'])
                p2.data.add_(grad2_scaled, alpha=-group['lr'])

class SGDv(Optimizer):
    def __init__(self, params, lr, weight_decay, betas=(0.9, 0.98), eps=1e-6, correct_bias=True):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)
    def step(self,closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])
                p.data.add_(p.grad.data, alpha=-group['lr'])
            

class AdamWr(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.0, correct_bias=False):
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
        # self.rank = rank
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
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p1.data)
                    state["exp_avg_sq"] = torch.zeros_like(p1.data)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1
                grad1 = p1.grad.data
                c = p2.data
                try:
                    c_ = torch.inverse(c.T@c)
                except:
                    c_ = torch.eye((c.T@c).shape[0]).to(c.device)
                grad1_scaled = c_@grad1
                assert grad1_scaled.shape == p1.grad.data.shape

                exp_avg.mul_(beta1).add_(grad1_scaled, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad1_scaled, grad1_scaled, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if 'correct_bias' in group and group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                c1 = p1.data
                if group["weight_decay"] > 0.0:
                    p1.data.add_(p1.data, alpha=-group["lr"] * group["weight_decay"])
                p1.data.addcdiv_(-step_size, exp_avg, denom)

                
                state = self.state[p2]
                
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p2.data)
                    state["exp_avg_sq"] = torch.zeros_like(p2.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                grad2 = p2.grad.data
                try:
                    c1_ = torch.inverse(c1@c1.T)
                except:
                    c1_ = torch.eye((c1@c1.T).shape[0]).to(c1.device)
                
                grad2_scaled = grad2@c1_
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