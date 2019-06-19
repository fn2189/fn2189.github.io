---
layout: post
title: Learning sorting functions on set: Order Matters
---

## Intro 1 (Abstract ?)


Since the rediscovery of automatic differentiation and its application to backpropagation in the 80s, machine learning and neural networks in particular have seen a huge resurgence. Various discoveries such as CNN or RNN and achievement such as the creation of the imageNet Dataset have made neural network relevant in a wide array of tasks. For example, in tasks such as image classification or in some settings of text-comprehension, neural network based algorithms have been shown to outperform humans. However, the scope of the type of problems for which neural networks are competitive is still relatively narrow and their performance can be underwhelming for problems for which we have otherwise found pretty efficient algorithms. Sorting problem are such problems. Although insertion sort for example gives us an algorithm to sort a list of integers in reasonable time,this type of problem are not trivially fit in one of the family of problems that neural networks have been show to work well (specifically classification or regression) and it is therefore not obvious how to try to solve them. Fortunately, building on tried and true recent ideas such as attention mechanism and pointer networks, the order matters network architecture came to the rescue. Coming up with a neural network architecture that can solve sorting problems has some interesting implications: current state of the art sorting algorithm require to know the order rule in place to be implemented. Conversely, because neural networks work by just learning a map between input and output pairs, that would not be required. As we will see, there are some interesting problems that can be reformulated as sorting problems for which the inherent order rule is seems extremely hard to grasp. 


## Intro 2

When dealing with problems where the input and output data are sequential, as is the case for neural machine translation tasks for example, most state of the art model are based on the sequence to sequence architecture. The information in the input sequence is summarized through an encoder into a fixed-size vector that is then used to initialize a decoder that p
predicts the output sequence. Both the encoder and the decoder use some kind of recurrent architecture to leverage the sequential nature of the input and output. Various extensions have been made to this basic architecture, most notably, the introduction of attention mechanisms that allows to prioritize the information from different steps of the input at each time step of the output and pointer networks, where the output at each step of the decoder comes from the input data and not from a fixed-size vocabulary.
With this improvements, neural machine translation tasks have reached human-level performance. However, when dealing with sorting problem over sets, the input data is no longer sequential in that the output sequence must be independent of the order in which the set is represented. The output data is the sorted sequence of elements of the input. One can still pick an order and treat this sorting problem as a sequence to sequence. However, as outlined in [Order Matters paper title], some input orderings might lead to sub par performance. (It amounts to trying to model a joint probability through different conditioning orders of the variables in a non convex optimization setting. The minima found will likely be local each time, some resulting in better performance than others). 
Making the encoder invariant to the order in which the input is represented is the focus of the order matters architecture (The paper actually also tackle the case where the output data is a set but that is not relevant for the sorting problem). In order to do so, the network is made of three blocks: a Read, Process and Write block. This architecture can then be tweaked to solve an array of sorting problem. In particular, I applied it for digits reordering, words reordering, an video reconstruction.

![The order Matters network architechture. ](https://raw.github.com/fn2189/fn2189.github.io/master/images/set_to_sequence.png "order matters network")


## Architecture

### Read block

The first block of the order matters network is the Read block. It's purpose is to independently encode each input to fixed-size vector representation. For the digits reordering problem, the input raw form can be seen as a 1-dimension vector. A suitable Read block is then an element-wise perceptron, to map the input to a higher dimension space, a special case of which could be simply the identity function. For the word reordering problem, using a character-level representation of the words as a sequence of #{vocab}-dimensional one-hot vector,  we use a LSTM cell to to encode each word.

```python
class ReadWordEncoder(nn.Module):
    def __init__(self, hidden_dim, input_size=26):
        super(ReadWordEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=1, batch_first=True)
```

```python
    def forward(self, x):
        l = []
        for i in range(x.size(0)):
            outputs, (h_n, c_n) =  self.lstm(x[i, :, :, :])
            l.append(h_n)
        res = torch.cat(l, dim=0).permute(0,2,1) #shape (batch_size, hidden_dim, n_set)
        return res
```

### Process block

The next block in the network is process block. It is where the encodings of the elements of the set are combined into a single, fixed-sized representation of the whole set. This is done using an LSTM cell without input combined with a Dot attention mechanism. Specifically, at step 0, a LSTM step is run with the input is initialized to 0 and the states are initialized randomly. The output q_t is used as a query for an a dot attention mechanism [(see more details on attention mechanism here)] to compute a relevance score e_i for each of the memories, which are the encoded inputs. Those relevance scores are then normalized through softmax to sum to 1. The resulting coefficient then used as the weights to compute the weighted sum of the memories that serves as the new input to the LSTM r_t while q_t serves as the new hidden state.


