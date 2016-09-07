# ispassive

Python package to determine if a sentence is using the passive voice.

## usage

The tagger needs to be trained before you can use it. This takes about a minute
and only has to be done once.

```python
In [1]: from ispassive import Tagger

In [2]: %time t = Tagger()

CPU times: user 45.9 s, sys: 347 ms, total: 46.3 s
Wall time: 46.3 s

In [3]: t.is_passive('Mistakes were made.')
Out[3]: True

In [4]: t.is_passive('I made mistakes')
Out[4]: False


```

## todo

Make it active voice.
