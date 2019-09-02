---
layout: post
title: Learning sorting functions on set: Order Matters
---
```diff
- We should make the title more visible, check the markdown specifications for that. 
```

Since the rediscovery of automatic differentiation and its application to backpropagation in the 80s, machine learning and neural networks in particular have seen a huge resurgence. In tasks such as image classification or in some settings of text-comprehension, neural network based algorithms have been shown to outperform humans. However, the scope of the type of problems for which neural networks are competitive is still relatively narrow and their performance can be underwhelming for problems for which we have otherwise found pretty efficient algorithms. Sorting problem are such problems. Although insertion sort for example gives us an algorithm to sort a list of integers in reasonable time, this type of problem are not trivially fit in one of the family of problems that neural networks have been shown to work well on (specifically classification or regression) and it is therefore not obvious how to try to solve them. Fortunately, building on recent ideas such as attention mechanisms and pointer networks, the Order Matters network architecture came to the rescue.

``` diff
- Put a reference here for the Order Matters paper. I would also add a few sentences introducing the paper, who worked on it, what their group is generally interested in. You should save this intro as it is for the CVPR paper (copy it into your draft), but break up this paragraph into two paragraphs with a little more history on this area of work for the blog. 
```

Coming up with a neural network architecture that can solve sorting problems has some interesting implications: current state of the art sorting algorithm require to know the order rule on the space elements are drawn from to be implemented. 
```diff
- That sentence is true, but hard to understand for novices. Try to break it down for a newbie
```
Conversely, because neural networks work by just learning a map between input and output pairs, that would not be required. As we will see, there are some interesting problems that can be reformulated as sorting problems for which the inherent order rule is seems extremely hard to grasp. 

When dealing with problems where the input and output data are sequential, as is the case for neural machine translation tasks for example, most state of the art models are based on the sequence to sequence architecture. The information in the input sequence is summarized through an encoder into a fixed-size vector that is then used to initialize a decoder that predicts the output sequence. Both the encoder and the decoder use some kind of recurrent architecture to leverage the sequential nature of the input and output. Various extensions have been made to this basic architecture, most notably, the introduction of attention mechanisms that allows us to prioritize the information from different steps of the input at each time step of the output and pointer networks, where the output at each step of the decoder comes from the input data and not from a fixed-size vocabulary.
With these improvements, neural machine translation tasks have reached human-level performance. However, when dealing with sorting problem over sets, the input data is no longer sequential in that the output sequence must be independent of the order in which the set is represented. The output data is the sorted sequence of elements of the input. One could still pick a given order to represent the input sequences and treat this sorting problem as a sequence to sequence one . However, as outlined in [Order Matters paper title], some input orderings might lead to sub par performance. 

```diff
- Add an annecdotal example of the kind fo sorting problem we have. Later, you will be doing number and word ordering, but at this point in the explanation, it would be useful to have a more difficult sorting problem. I with thinking putting dinner guests in the correct order at a dinner party. Like if there was a State dinner and the dignitaries of the world had to be put in the correct order when seated around the table. The guests are a 'set' of items and therre is a known 'best order' for their seatting arrangement, but the ordering function is quite difficult to describe. 

- This is a cool and complicated aside! I would definitely expand it for the CVPR paper version. 
```
(It amounts to trying to model a joint probability through different conditioning orders of the variables in a non convex optimization setting. The minima found will likely be local each time, some resulting in better performance than others). 
Making the encoder invariant to the order in which the input is represented is the focus of the order matters architecture

```diff
- I would mention this aside in the CVPR paper in the Related Work section, but it can be skipped in this blog
```
(The paper actually also tackle the case where the output data is a set but that is not relevant for the sorting problem). In order to do so, the network is made of three blocks: a Read, Process and Write block. This architecture can then be tweaked to solve an array of sorting problem. In particular, I applied it for digits reordering and word reordering.

[//]: # (, an video reconstruction.)


## Architecture
![The order Matters network architechture. ](https://raw.github.com/fn2189/fn2189.github.io/master/images/set_to_sequence.png "order matters network")
```diff
- We need a caption here
```

### Read block

The first block of the order matters network is the Read block. It's purpose is to independently encode each input to fixed-size vector representation. For the digits reordering problem, the input raw form can be seen as a 1-dimension vector. A suitable Read block is then an element-wise perceptron, to map the input to a higher dimension space, a special case of which could be simply the identity function. For the word reordering problem, using a character-level representation of the words as a sequence of #{vocab}-dimensional one-hot vector,  we use a LSTM cell 

```diff
- Can you make the width of the text wider? So the code below is more readable on a laptop/ desktop screen?
```

```python
class ReadWordEncoder(nn.Module):
    def __init__(self, hidden_dim, input_size=26):
        super(ReadWordEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=1, batch_first=True)
```

to to encode each word. With x of shape (batch_size, n_set, max_word_length, vocab_size), for each set in the batch, we treat x(i,:,:,:) as an 3-D input of batch_size n_set to the aforementionned LSTM . We use the last hidden_state of the LSTM, of shape (n_set, hidden_dim) as the representation the set

```python
    def forward(self, x):
        """
        x: shape (batch_size, n_set, max_word_length, vocab_size)
        """
        l = []
        for i in range(x.size(0)):
            outputs, (h_n, c_n) =  self.lstm(x[i, :, :, :])
            l.append(h_n)
        res = torch.cat(l, dim=0).permute(0,2,1) #shape (batch_size, hidden_dim, n_set)
        return res
```
After concatenating the representaions for all the sets in the batch and reordering the dimensions, we get an output of shape (batch_size, hidden_dim, n_set)

```diff
- Han this part is for you! Let's make this look nice
```

![Read Block ](https://raw.github.com/fn2189/fn2189.github.io/master/images/read_block.png "Read Block")
*The Architecture of a read block*

### Process block

The next block in the network is process block. It is where the encodings of the elements of the set are combined into a single, fixed-sized representation of the whole set. This is done using an LSTM cell without any input combined with a Dot attention mechanism. 

```diff
- Put the citation for this figure in the caption. We might want to redraw it to match the other figures even.
```

![An Attention Mechanism. ](https://raw.github.com/fn2189/fn2189.github.io/master/images/attention%20mechanism.png "An attention Mechanism")

Specifically, at step 0, a LSTM step is run with the input is initialized to 0 and the states are initialized randomly. 

```python
class Process(nn.Module):
    """..."""
    def __init__(self, input_dim, hidden_dim, lstm_steps, batch_size):
        """..."""
    def forward(self, M, mask=None, dropout=None):
        """..."""
        #To account for the last batch that might not have the same length as the rest
        batch_size = M.size(0)
        i0 = self.i0.unsqueeze(0).expand(batch_size, -1)
        h_0 = self.h_0.unsqueeze(0).expand(batch_size, -1)
        c_0 = self.c_0.unsqueeze(0).expand(batch_size, -1)
        
        for _ in range(self.lstm_steps):
            if _ == 0:
                h_t_1 = h_0
                c_t_1 = c_0
                r_t_1 = i0
            h_t, c_t = self.lstmcell(r_t_1, (h_t_1, c_t_1))
```
```diff
- we should figure out how to show mathematical notation (q_t) more nicely in markdown. 
```

The output q_t is used as a query for an a dot attention mechanism [(see more details on attention mechanism here)] to compute a relevance score e_i for each of the memories M, which are the encoded inputs. 


```python
        scores = torch.matmul(M.transpose(-2, -1), c_t.unsqueeze(2)) \
                         / math.sqrt(d_k)
```
Those relevance scores are normalized through softmax to sum to 1. 

```python
        p_attn = F.softmax(scores, dim = -1)
```

The resulting coefficients then used as the weights to compute the weighted sum of the memories that serves as the new input to the LSTM r_t while q_t serves as the new hidden state.
```python
        r_t_1 = torch.matmul(M, p_attn).squeeze(-1)
            #print(f'r_t_1: {r_t_1.size()}')
            h_t_1 = h_t
            c_t_1 = c_t
```

```diff
- Han this part is for you! Let's make this look nice
```
![Process Block ](https://raw.github.com/fn2189/fn2189.github.io/master/images/process_block.png "Process Block")
*The Architecture of a process block*

### Write Block

The last block is the write block, in charge of outputting a sequence representing the correct order of the element in the cell.

This idea is similar to the decoder part of a sequence to sequence architecture except that instead of outputting a element from a vocabulary at each step, an element of the input set is outputted, or better so, pointed to. To achieve that, like for the process block, an LSTM with dot attention over the input memories is used but instead of using the normalized relevance coefficients in a weighted sum, they are used as probabilities and the most likely element is outputted at each step. 
```python
    # Recurrence loop
    for _ in range(input_length):
        h_t, c_t, outs = step(decoder_input, hidden)
        hidden = (h_t, c_t)
        # Masking selected inputs
        masked_outs = outs * mask

        # Get maximum probabilities and indices
        max_probs, indices = masked_outs.max(1)
        one_hot_pointers = (runner == indices.unsqueeze(1).expand(-1, outs.size()[1])).float()

        # Update mask to ignore seen indices
        mask  = mask * (1 - one_hot_pointers)

```

In order to avoid the network repeating itself by outputting the same element multiple time, elements are masked out once they are outputted the first time. 
```python
        # Update mask to ignore seen indices
        mask  = mask * (1 - one_hot_pointers)
```

This decoder architecture is fittingly called a pointer network.

```diff
- Franck, can you redo the below figure like you did the process block? I think it will be easier for Han to make it look nice if you've already re-worked it a bit.
```

![Pointer Network ](https://raw.github.com/fn2189/fn2189.github.io/master/images/pointer%20networks.png "Pointer Network")

## Experiments

As Mentionned previously, we tried to apply the order matters architecture to 3 difffent problems of (according to us) increasing complexity: digits, words and video reordering. This allowed us to see how the architecture performs in setting where the sorting problem is easily solved by humans because the underlying order is obvious (digits and words) and one that might have hard instances for humans because said underlying order in not trivial. Forthe digits and word reordering problems, we experimented with the set size to try to expose the limits of our implementation but we did not for video because the problem was hard enough to solve with a set size of 5.

### Digits 
``` diff
- Add training/ validataion/ test trends
- Add annecdotal examples of well done tricky sorting and any examples of failure cases
```
#### Dataset
The dataset for this problem can be generated , which allow us a lot of room for experimentation. We just pick a set size as well as train, val and test sizes and genrerates pair of X (n_set floats) and Y(lis of the indexes that would sort the set.

#### Settings and parameters

As mentionned previously, the read block is perceptron. We choose a single layer with a hidden dimension of 32 and a ReLu activation. This hidden dim is also used for the process and write block. We use Adam optimizer with a learning rate of 1e-4, a batch size of 256 and no dropout.

#### Metric and Result
The loss used is an element-wise cross-entropy loss but the accuracy is measured by a 1-0 loss (1 if the predicted and correct order are exactly equal and 0 otherwise. This is a rather conservative metric because it only rewards exact matches. On our validation set, we get a perfct accuracy.

### Words
``` diff
- Add training/ validataion/ test trends
- Add annecdotal examples of well done tricky sorting and any examples of failure cases
```
#### Dataset
The dataset for this problem can be either synthetic or with word coming from an or many existing dictionaries. Those 2 aproaces are sligthy different because the probability space of the words changes. We  opted for the most inteesting setting where te training data is generated from a uniform distribution ofthe word of between 5 and 26 letters from the english alphabet while the testing is done on words coming from the english dictionary. Either way, a word is represented as a matrix of one-hot-encoded characters. The dataset creation is otherwise similar to the digits reordering problem

#### Settings and parameters

As mentionned previously, the read block is a char-level RNN using an LSTM. The vocabulary is of size 26 (size of the english alphabet without any special character). The hiddendim of the LSTM is of size 32. This hidden dim is also used for the process and write block. We use Adam optimizer with a learning rate of 1e-4, a batch size of 256 and no dropout.

#### Metric and Result
The training loss and accuracy metric used are exactly the same as for the word reordering problem. On our validation set, we get a perfct accuracy.

[//]: # (### Videos

[//]: # (#### Dataset
[//]: # (There are a few things we had to consider when picking a dataset for this problem. Because most state-of-the-art feature computation algorithms for video leverages pretrained image classiffication neural nets to compute frame-level features (obtained for example for an earlier fully connected layer of a NN) that are then combined to get a feature representatioon of the whole video. That means that we need to pick a video dataset and a pretrained network. How close the concepts present in the videsos are the the one of the last layer of the pretrained NN might affect the quality of the feature representation. In thi case, we sed a pretrained mobilenet v2 as a feature extractor and a proprietary video dataset. To build the sets, we chop the videos in n_set segments that are randomly shuffled. The input X is then a matrix of feature representations for each of the segments in the set while Y is the list of indexes that would sort the set back to the original order of the video.)

[//]: # (#### Settings and parameters

[//]: # (The read block is here again a single layerperceptron with a ReLu nonlinearity that map the feature space from the NN feature representation to a new feature space. We use a hidden dim of 512. This hidden dim is also used for the process and write block. We use Adam optimizer with a learning rate of 1e-4, a batch size of 256 and a dropout rate of .2, mainly in the read block.

[//]: # (#### Metric and Result
[//]: # (The training loss and accuracy metric used are exactly the same as for the word reordering problem. So far, on our validation set, we get an accuracy of about 0.09, which is definitely not great but already better that random. The accuracy on the training set in significantly higher, so the network is overfitting. We have not yet been able to successfully overcome this problemat the time of the writing.



## Next Steps
```diff
- Next steps are now getting this to work for video, but you don't have to go into much detail about that.
- It would be better just to have a paragraph like "It's it cool what we did with Order Matters?!"
```

For multiple tasks such as videos classification, optical flow features have been shown to improve performance. It seems reasonnable to assume that they can convey some meaningful informaion about how to sort the segments of a video so it would be interesting to include optical flow in the feature represntation of the videos segments to see if it helps solving the sorting problem.



