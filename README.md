Makemore is a language model that takes in training data of lines of words and outputs words that are similar

- bigram.ipynb uses a bigram language model to construct new names. First, constructing a table of bigram counts from names in names.txt. This model then randomly selects the next character using the multinomial probability of each next character based on the previous character. This is also done using a sigle layer neural network using PyTorch. Results are simliar, because the layer of the neural net is essentially trained to be the same as the table of bigrams from the previous part.

Followed Andrej Karpathy's Youtube tutorial on his Youtube channel
