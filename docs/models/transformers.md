---
title: Transformers
layout: default
nav_order: 5
parent: Models
mathjax: true
---

# Transformers 
<img src="{{ site.baseurl }}/assets/images/transformer_icon.jpg" alt="Transformer Icon" width="50" style="float:right"/>

{: .no_toc }

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

## Introduction

The Transformer architecture has become the universal powerhorse of modern AI. It drives everything from the immense reasoning capabilities of Large Language Models to the precision required for Google Photos to identify a "cute kitten pictures" in your library.  
On this page you will learn the core mechanisms behind transformer network and how it became the universal backbone for any deep learning task.  
Before reading this page however, we strongly recommend you to check and learn how the **[Convolutional Networks]({{ site.baseurl }}{% link docs/models/convolutional-networks.md %})** works.

## Building the transformer block by block
### Input Block
<div align="center">
    <img src="{{ site.baseurl }}/assets/images/input_block.png" alt="Input Block" style="display: block; margin: auto; width: 70%;"/><figcaption><i>Input block of the transformer.</i></figcaption>
</div>

#### Input (word) Embeddings 
To understand how a Transformer thinks, it is helpful to develop a geometrical intuition of how it represents information. Inside the model, language is not stored as letters or strings, but as tokens—the fundamental building blocks of text.

While a token usually represents a group of characters or sub-words (rather than a single whole word), for the sake of simplicity, we can think of each token as an individual word. For example the sentence:
<p align="center">
  <i>"UROB is the best course in the world!"</i>
</p>
Can be tokenized as:
<p align="center">
  <i>"UROB", "is", "the", "best", "course", "in", "the", "world", "!"</i>
</p>

These tokens live in a high-dimensional space. The process known as <b>Embedding</b> transforms a token into a numerical vector that carries specific geometric meaning.
- <b>Semantic proximity:</b> Tokens with similar meanings or grammatical roles are close to each other.
- <b>Logic as direction:</b>Distance and angle between the individual embeddings represents logical and contextual relationship between the words.  
<small><i>**Btw, one of the grandfather of modern embeddings was <a href="https://en.wikipedia.org/wiki/Word2vec" target="_blank">Word2Vec</a> developed by a Czech researcher <a href ="https://en.wikipedia.org/wiki/Tom%C3%A1%C5%A1_Mikolov" target="_blank">Tomas Mikolov</a></i></small>

<div align="center">
    <img src="{{ site.baseurl }}/assets/images/word_embedding.png" alt="Word Embeddings" style="display: block; margin: auto; width: 70%;"/><figcaption><i>Geometrical representation of word embeddings​.</i></figcaption>
     <i>Notice the operation "king" + "woman" = "queen".</i>
</div>

#### Positional Encoding
Since the transformer architecture does not have any built-in notion of order (unlike RNNs or CNNs), we need to provide some information about the position of each token in the sequence. This is done using <b>Positional Encoding</b>, which adds a unique vector to each token embedding based on its position in the sequence. This way, the model can differentiate between tokens based on their order.  
The positional encoding can be passed as both static (original transformer from <a href="https://arxiv.org/abs/1706.03762" target="_blank">Attention Is All You Need</a>) or learnable vectors (GPT).


### Self-Attention Block
Now that we have our input embeddings ready, it's time to understand the core mechanism that makes transformers so powerful: the Attention Mechanism.
<div align="center">
    <img src="{{ site.baseurl }}/assets/images/self_attention.png" alt="Attention Block" style="display: block; margin: auto; width: 70%;"/><figcaption><i>Attention block of the transformer.</i></figcaption>
</div>

From the input embeddings, we derive three crucial components: Queries (Q), Keys (K), and Values (V). These are obtained by multiplying the input embeddings with three different weight matrices that are learned during training.
These new components have the following roles:  
- **Query (Q):** "What am I looking for?" Represents the "search criteria" of the current token. It probes all other tokens to see which ones are relevant to its context.

- **Key (K):** "How do i Identify myself?" This vector represents the "identity" of each token in the sequence. It is used to match against the query <b>Q</b>. It can be thought of as potential for relevevance to different queries.

- **Value (V):** "What is my actual content?" This vector contains the actual information that will be passed as the query <b>Q</b> finds a matching key <b>K</b>.

Another analogy to understand these components is to think of them as a library system:
- The **Query** "What you type into search bar (History of Rome...)".
- The **Key** "The index cards in the library catalog (Rome, History, Ancient...)".
- The **Value** "The actual books on the shelves (The Rise and Fall of the Roman Empire...)".



Now the next natural move is to compare the Query <b>Q</b> with all Keys <b>K</b> to determine how relevant each token is to the current token's context. This is done by calculating the attention score using the following formula:
$$\text{Attention Score} = {Q \cdot K^T}$$

Now we can use these attention scores to weigh the Values <b>V</b> and mix them to get the context-aware representation of the current token <b>Z</b>: 
<div align="center">
  
$$Z = \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right)V$$

<small>
Note that the attention scores are often scaled by the square root of the dimension of the keys ($d_k$) to prevent extremely large values that can destabilize the softmax function.
</small>
</div>

For example the word bank is very ambigouous. Without context it is impossible to determine if it refers to a financial institution or the side of a river. Before the attention mechanism, both menanings are equally likely and hence the embedding of the word bank would be somewhere in the middle of the "money" and "river" clusters. With the attention mechanism the other words in the sentence will impact the embedding of the word bank and move it closer to the actual meaning in the sentence.

<div align="center">
  <i>You can get an <span style="color: #E69F00">insurance</span> in a bank.</i> <br>
  <i>The <span style="color: #56B4E9">river</span> bank was <span style="color: #56B4E9">flooded</span>.</i> <br>

  <small>Context helps distinguish the meanings!</small>
</div>


This is the essence of the self-attention mechanism: each token can "attend" to all other tokens in the sequence, allowing the model to capture complex dependencies and relationships.

### Multi-Head Attention

<div align="center">
    <img src="{{ site.baseurl }}/assets/images/multihead_attention.png" alt="Multi-Head Attention" style="display: block; margin: auto; width: 70%;"/><figcaption><i>Multi-Head Attention block</i></figcaption>
</div>

To enhance the model's ability to capture different types of relationships, transformers use a technique called <b>Multi-Head Attention</b>. Instead of performing a single attention operation, the model splits the input into multiple "heads", each with its own set of Q, K, and V matrices. This allows the model to focus on different aspects of the input simultaneously. These heads <b>represent different subspaces </b> of the input data and are smaller in dimension compared to the original embedding size. The outputs of all heads are then concatenated to produce the final output.  
The benefit of this approach is that instead of trying to learn everything in a single attention mechanism, the model can learn to <b>focus on different features or relationship at once</b>, leading to a richer understanding of the input.

### Building the Transformer
Now that we understand the self-attention mechansim, we can build the full transformer block.