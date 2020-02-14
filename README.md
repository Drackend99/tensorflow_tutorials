# Tensorflow tutorials

As you go, be sure to install any dependencies you're missing with pip3, since we'll be using python 3

## Table of Contents

# Tutorial 1: Text Generation with LSTMs

## Step 0: Imports

We're going to need some libraries. We may or may not add more as we go, but to start with we will need the following:

```
import tensorflow as tf
from tensorflow import keras
import numpy as np
```

## Step 1: Load in data

Tensorflow has lots of test datasets available for us, but we're going to use one of our own: Treasure Island. Our network will read this book, and then be able to generate text based on what it's learned. Treasure island is free to get since it's so old.

To start, add ```import requests``` to your imports. ```requests``` is a standard python library (no extra installation needed) that allows us to get data from an https request. In other words, input a url, and it will get the text from that url.

```
data = requests.get("https://data.heatonresearch.com/data/t81-558/text/treasure_island.txt")
```

We're going to want to get the text from her, rather than the raw byte data, so lets go ahead and covert that:

```
raw_data = data.text
```

## Step 2: Preprocess

Preprocessing is a step that varies on the basis of what project you're doing. Here, we'll want to break up our corpus of text into a form that allows us to gather information about the words it contains.

Our first step is to make everything lower case. Why? It's easier.

```
preprocessed_text = raw_data.lower()
```

Next, we want to get rid of anything that isn't ascii (standard keyboard alphabet). It'll just confuse the computer, and us for that matter. 

We can do so with **regular expressions**, which are a way of searching for particular sequences within a string. Basically, they're substrings on steroids. The downside is their syntax makes no sense at first glance, but the upside is you can really just look up what to put without needing to memorize it. We can do this using ```import re```.

Specifically, we'll be using ```re.sub(text_to_replace, replacement, source)```, which takes what it finds and subtracts it from the original string. This will get rid of our non-ascii values.

```
preprocessed_text = re.sub(r'[^\x00-\x7f]', r'', preprocessed_text)
```

Don't worry about the syntax for that. When you go to make your own program, you can just figure out what you need.

Continuing on, we want to extract the characters of the text, sort them, and assign an ID to them. By doing this, we can have an organized record of each character. 

Our plan to do this is as follows:
1. Cast ```preprocessed_text``` into a set. 
    - A set is a list of **unique** elements; an element will not be added to a set if the set already contains it will not be added.
    - What this means for our text is that each of the characters will be added to the set. If it's already seen the character, it won't add it to the set. Thus, we'll get a master list of the characters that appear in the text
2. Cast ```set(preprocessed_text)``` into a list.
    - As useful as sets are to fill a list, once we've filled it, we'd rather work with a list.
3. Sort our ```list(set(preprocessed_text))```
    - Like I said, we want the IDs to be organized. We had better sort our resulting list.

```
chars_set = set(preprocessed_text)
chars_list = list(set(preprocessed_text))
chars_list_sorted = sorted(chars_list)
```
Note we use ```sorted(chars_list)``` and not ```chars_list.sort()```. The ```.sort()``` method is *insitu*, meaning it does not return anything and sorts the list in place. We can't set a new variable equal to a ```list.sort()```, because the new variable would just be assigned ```None```.

Now that we've got our list of characters that appear in the text, we can form 2 dictionaries: one for converting a character into its ID, and one for converting an ID into its character.

We can use some advanced python to make this nice and simple, through ```enumerate```. As you may know, python has two types of for loops: 
1. One for iterating through items: ```for item in list```
2. One for iterating through indices: ```for i in len(list)```

```Enumerate``` simply combines the two: ```for i, item in enumerate(list)```

One more concept to make this even easier is called **list comprehension**. You've probably seen it before:
```list = [x for i in range(5)]```
All this does is allow you to fill a list (or any other collection) without needing to write out the whole for loop. The above command is equivelant to:
```
list = []
for i in range(5):
    list.append(i)
```
Please don't use list as your variable name though. That'll cause you many problems. Anyway, back to the task at hand.

We can combine these two concepts to fill our dictionaries. We can use the ```dict()``` function to create it:

```
char2index = dict((char, index) for char, index in enumerate(chars_list_sorted))
index2char = dict((index, char) for char, index in enumerate(chars_list_sorted))
```

We defined the ```(key, value)``` of our dictionary to be either ```(char, index)``` or ```(index, char)```. We fill it by iterating through our ```char_list_sorted```.

## Step 3: Generating our data

Now that we've defined our "dictionary" (literally!) of characters, we can build sequences of these characters.

In standard neural networks, we have x (input) and y (output). However, here we will be using something called **LSTM**, or long short-term memory. For LSTMs, x and y will be **sequences**. The x input will specify the sequences where the y sequences are the expected output.

First, we will need to define our sequence size. We will be grabbing 40-character blocks:
```
max_length = 40
```
Our program will be given 40 characters, and try to guess the next character. We'll use 40 *real* characters as a seed to start our newly-generated article.

We'll also set a step size, equal to 3. This means the for every 40-character block you grab, move forward 3 charcters. In other words, there will be 37-character overlap between each block. This small step size allows for better learning of context.

```
step_size = 3
```
From here, we can begin to generate our sequences. We'll make two lists to fill.
- ```sentences```: The block of 40 characters
- ```next_char```: The letter that the current block ends right before. It will represent the difference between this block and the next, but we'll get more into that later.

```
sentences = []
next_char = []
for i in range(0, len(preprocessed_text) - max_length, step_size):
    sentences.append(preprocessed_text[i:i+max_length])
    next_chars.append(preprocessed_text[i+max_length])
```

For treasure island, this gives us 132454 blocks of 40 characters. Plenty of training data.

Our final step is to vectorize our data, which chagnes it into the proper format that our network needs. 

We can start by defining some blank vectors. Our input vector ```x``` will be three dimensional:
- Size 1D: Amount of data. Each index will be the index of a piece of data.
- Size 2D: Size of input vector. Each index will be the data of the piece.
- Size 3D: # of possible characters. This will act as a "dummy variable", for the next possible character.

What's a "dummy variable"? I'm glad you asked. Categorical data isn't quantitative. However, our vectors can only have numbers. Thus, it's up to us to make it quantitative. A "1" indicates that the categorical feature is present, while a "0" indicates that it isn't. 

For us, the categorical features are characters. And, since only one character can be present in each character slot, we'll only ever have one "1", and 59 "0"s.

Our output vector ```y``` will only by two dimensional, representing the index of a data piece and its output, not needing the input itself.

```
x = np.zeros((len(sentences), max_length, len(chars_list_sorted)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
```

Then, we can fill in these vectors with the actual data of what characters are present:

```
for i, sentence in enumerate(sentences):
    for t, char in enumerate(chars_list_sorted):
        x[i, t, char2index[char]] = 1
    y[i, char2index[next_char[i]]] = 1
```

## Step 4: Building the network

Now with all that out of the way, we can get to building the network. Like for a standard neural network, we'll be using tensorflow's sequential class, which just means a sequential (linear) stack of layers (e.g. input -> hidden -> hidden -> output).

```
model = tf.keras.Sequential()
```

We'll be using LSTMs
