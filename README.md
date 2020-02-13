# Tensorflow tutorials

As you go, be sure to install any dependencies you're missing with pip3, since we'll be using python 3

## Table of Contents

# Tutorial 1: Text Generation

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
preprocessed_text = re.sub(r'\x00-\x7f]', r'', preprocessed_text)
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
chars_list_sorted = chars_list.sort()
```
