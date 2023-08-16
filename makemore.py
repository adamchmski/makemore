import torch
import torch.nn.functional as F
import sys
import random
import tqdm
import argparse 

parser = argparse.ArgumentParser(prog='Makemore', description='Takes an input of a list of words and outputs similar words')
parser.add_argument("-f", "--filename", type=str, default="names.txt", metavar="path/to/file", help="The file that contains the training data")
parser.add_argument('-n', '--training_iterations', type=int, default=1000, metavar=">0", help="Sets the number of NN training iterations for the selected model")
parser.add_argument('-m', '--model', type=str, default="bigram", choices=['bigram', 'mlp'], metavar="[bigram, mlp, ]", help="Choose the NN model to use")
parser.add_argument('-s', '--sample', type=int, default=20, metavar="int of output samples", help="Sets the amount of output samples")

args = parser.parse_args()
file_path = args.filename
training_iterations = args.training_iterations
model = args.model
num_samples = args.sample

if training_iterations < 1:
    print("Error: Training iterations must be >= 1")
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

def MLP():
    def build_dataset(words):
        block_size = 3 
        X, Y = [], []
        
        for word in words:
            context = [0] * block_size
            
            for ch in word + '.':
                X.append(context)
                ix = stoi[ch]
                Y.append(ix)
                context = context[1:] + [ix]
        
        return torch.tensor(X), torch.tensor(Y)

    random.shuffle(words)
    n1 = int(.8 * len(words))
    n2 = int(.9 * len(words)) 

    # create dataset of tensors 
    Xtrn, Ytrn = build_dataset(words[:n1])
    Xdev, Ydev = build_dataset(words[n1:n2])
    Xtst, Ytst = build_dataset(words[n2:])

    #initialize weights and biases
    C = torch.randn((27,10))
    W1 = torch.randn((30, 200)) * 0.1
    b1 = torch.randn((200)) * 0.1
    W2 = torch.randn((200, 27)) * 0.1
    b2 = torch.randn((27)) * 0.1
    parameters = [C, W1, b1, W2, b2]
    for p in parameters:
        p.requires_grad = True

    # train the neural net 
    loss_val = 0

    for _ in tqdm.tqdm(range(training_iterations), desc="Processing", ncols=100):

        mini_batch_ix = torch.randint(0, Xtrn.shape[0], (32,))

        # forward pass
        emb = C[Xtrn[mini_batch_ix]]
        h = torch.tanh(emb.view(-1,30) @ W1 + b1)
        logits = h @ W2 + b2 
        loss = F.cross_entropy(logits, Ytrn[mini_batch_ix])
        loss_val = loss.item()
        
        # backward pass
        for p in parameters:
            p.grad = None
        loss.backward()

        # update
        for p in parameters: 
            p.data += -0.01 * p.grad
    
    print("Loss is", loss_val)

    # Output samples 
    block_size = 3
    for _ in range(num_samples):

        context = [0] * block_size
        out = []
        ix = 0
        while True:
            emb = C[torch.tensor([context])]
            h = torch.tanh(emb.view(1,-1) @ W1 + b1)
            logits = h @ W2 + b2 
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, replacement=True).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break
        print(''.join(itos[i] for i in out))


# Message before new words are outputted 
print(f"\nNewly created words based off of {file_path.split('/')[-1]} \n")

# Select the model 
if model == 'bigram':
    bigram_model()
elif model == 'mlp':
    MLP()