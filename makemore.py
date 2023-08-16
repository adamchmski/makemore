import torch
import torch.nn.functional as F
import sys
import random
import tqdm
import argparse 

parser = argparse.ArgumentParser(prog='Makemore', description='Takes an input of a list of words and outputs similar words')
parser.add_argument("-f", "--filename", type=str, default="names.txt", metavar="path/to/file", help="The file that contains the training data")
parser.add_argument('-n', '--training_iterations', type=int, default=1000, metavar="[100-10,000]", help="Sets the number of NN training iterations for the selected model")
parser.add_argument('-m', '--model', type=str, default="bigram", choices=['bigram', 'mlp'], metavar="[bigram, mlp, ]", help="Choose the NN model to use")
parser.add_argument('-s', '--sample', type=int, default=20, metavar="int of output samples", help="Sets the amount of output samples")

args = parser.parse_args()
file_path = args.filename
training_iterations = args.training_iterations
model = args.model
num_samples = args.sample

if training_iterations > 10000 or training_iterations < 1:
    print("Error: Training iterations must be in range (1-10,000)")
    sys.exit()

content = ""

# Read in the text file 
try:
    with open(file_path, 'r') as file:
        content = file.read()
except FileNotFoundError:
    print("File not found.")
    sys.exit() 

except Exception as e:
    print("An error occurred", e)
    sys.exit() 

# Create a mapping to/from characters and integers  
words = content.split()
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i:s for s, i in stoi.items()}


# BIGRAM NEURAL NET 
def bigram_model():

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
    W = torch.randn((27,27), requires_grad=True)

    num = len(xs)
    loss_val = 0

    # Gradient descent 
    for _ in tqdm.tqdm(range(training_iterations), desc="Processing", ncols=100):
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

    # Creating names with the neural net 
    print(f"\nNewly created words based off of {file_path.split('/')[-1]} \n")
    for i in range(num_samples):
        ix = 0
        out = []
        
        while True:
            xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
            logits = xenc @ W
            counts = logits.exp()
            probs = counts / counts.sum(1, keepdims=True)
            ix = torch.multinomial(probs, num_samples=1, replacement=True).item()
            out.append(itos[ix])
            if ix == 0:
                break
        print(''.join(out))  



# Select the model 
if model == 'bigram':
    bigram_model()
elif model == 'mlp':
    pass