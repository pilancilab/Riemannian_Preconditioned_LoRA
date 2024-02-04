import torch
import matplotlib.pyplot as plt 
import pickle
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

def ema(scalars, weight=0.99):  
        last = scalars[0]  
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point  
            smoothed.append(smoothed_val)                        
            last = smoothed_val                                  
        return smoothed
    
def avg(scalars):
    c = []
    for i in range(len(scalars)):
        c.append(sum(scalars[:(i+1)])/(i+1))
    return c

with open('../LoRA_SGD_1e4_trainloss.pickle', 'rb') as f:
    lora_sgd_1e4 = pickle.load(f)

with open('../LoRA_SGD_1e3_trainloss.pickle', 'rb') as f:
    lora_sgd_1e3 = pickle.load(f)

with open('../LoRA_SGD_1e2_trainloss.pickle', 'rb') as f:
    lora_sgd_1e2 = pickle.load(f)

with open('../LoRA_reSGD_1e4_trainloss.pickle', 'rb') as f:
    lora_resgd_1e4 = pickle.load(f)

with open('../LoRA_reSGD_1e3_trainloss.pickle', 'rb') as f:
    lora_resgd_1e3 = pickle.load(f)

with open('../LoRA_reSGD_1e2_trainloss.pickle', 'rb') as f:
    lora_resgd_1e2 = pickle.load(f)



sgd_1e2 = torch.Tensor(avg(lora_sgd_1e2))
sgd_1e3 = torch.Tensor(avg(lora_sgd_1e3))
sgd_1e4 = torch.Tensor(avg(lora_sgd_1e4))
resgd_1e2 = torch.Tensor(avg(lora_resgd_1e2))
resgd_1e3 = torch.Tensor(avg(lora_resgd_1e3))
resgd_1e4 = torch.Tensor(avg(lora_resgd_1e4))
stack_data = torch.stack([sgd_1e2,sgd_1e3,sgd_1e4]).T
df = pd.DataFrame(stack_data, columns=['SGD with lr=1e-2','SGD with lr=1e-3','SGD with lr=1e-4'])
sns.set_palette("flare")
sns.lineplot(data=df,dashes=False,linewidth=5)
sns.set_palette("tab20c")
stack_data = torch.stack([resgd_1e2,resgd_1e3,resgd_1e4]).T
df = pd.DataFrame(stack_data, columns=['scaled GD with lr=1e-2','scaled GD with lr=1e-3','scaled GD with lr=1e-4'])
sns.lineplot(data=df,dashes=False,linewidth=5)
plt.legend(fontsize=18)
plt.xlabel('# iterations', fontsize=30)
plt.ylabel('training loss', fontsize=30)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()
plt.savefig('loss.pdf')
plt.clf()
