# Makemore 
Makemore is a character level language model that takes in training data of lines of words and outputs words that are similar

### Usage
The following is the most basic way to run the program 

`python3 makemore.py`

Which defaults to 

`python3 makemore.py -f=names.txt -n=1000 -m=bigram -s=20`

#### Optional Arguments:

show this help message and exit: `-h, --help` 


The file that contains the training data: `-f path/to/file, --filename path/to/file`


Sets the number of NN training iterations for the selected model: `-n > 0, --training_iterations > 0`


Choose the NN model to use: `-m [bigram, mlp, ], --model [bigram, mlp, ]`

  
Sets the amount of output samples: `-s int of output samples, --sample int of output samples`


### Models 
- names.txt contains over 32,000 names. Used for training data.  

- bigram uses a bigram language model to construct new names. First, constructing a table of bigram counts from names in names.txt. This model then randomly selects the next character using the multinomial probability of each next character based on the previous character. This is also done using a sigle layer neural network using PyTorch. Results are simliar, because the layer of the neural net is essentially trained to be the same as the table of bigrams from the previous part.

- mlp uses a multi-layer perceptron to predict the next characer with the last 3 characters. Everything is done manually except for backpropogation. The MLP approach leads to better results than the bigram model by preventing single letter names and other strange outputs that the bigram model output.  

Followed Andrej Karpathy's Youtube tutorial on his Youtube channel
