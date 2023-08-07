Makemore is a language model that takes in training data of lines of words and outputs words that are similar

bigram.ipynb uses a bigram language model to construct words. First using a table of bigrams, constructed from names in names.txtthat contains the counts of each bigram. This model selects a next character using the multinomial probability of each next character based on the previous character. Next, this is done with a sigle layer neural network using PyTorch. Results are simliar, because the layer of the neural net is essentially trained to be the same as the table of bigrams from the previous part.

Followed Andrej Karpathy's Youtube tutorial on his Youtube channel
