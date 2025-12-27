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
#### Word Embeddings 
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
