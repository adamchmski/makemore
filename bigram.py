import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys
import random
import tqdm
import argparse 

#parser = argparse.ArgumentParser(prog='Makemore', description='Takes an input of a list of words and outputs similar words')

file_path = sys.argv[1]
content = ""

# Read in the text file 
try:
    with open(file_path, 'r') as file:
        content = file.read()
except FileNotFoundError:
    print("File not found.")
    sys.exit()  # This will terminate the program

except Exception as e:
    print("An error occurred", e)
    sys.exit()  # This will terminate the program

# Create a mapping to/from characters and integers  
words = content.split()
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i:s for s, i in stoi.items()}

# BIGRAM NEURAL NET 

# Create training set 
xs, ys = [], []

for word in words:
    chs = ["."] + list(word) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        i1 = stoi[ch1]
        i2 = stoi[ch2]
        xs.append(i1)
        ys.append(i2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

# Train neural net
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27,27), generator=g, requires_grad=True)

num = len(xs)
loss_val = 0

# Gradient descent 
for _ in tqdm.tqdm(range(1000), desc="Processing", ncols=100):
    # forward pass
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True)
    loss = -probs[torch.arange(num), ys].log().mean()
    loss_val = loss.item()
    
    # backward pass 
    W.grad = None 
    loss.backward()
    W.data += -10 * W.grad
print(loss_val)

# Creating names with the neural net 
print("Newly created names based off of ")
for i in range(20):
    ix = 0
    out = []
    
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdims=True)
        ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))  