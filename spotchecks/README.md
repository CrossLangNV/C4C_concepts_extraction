# Potential code issues
Please do not panic or stress out. I just spotted some potential issues in your code.
I know that this is old code, so by now you probably know the answer to the following questions:

## 1
In the following code, you are mixing a translation table with cascaded substitutions.
- Why is this not scalable?
- Why is this hard to maintain?
```python
def clean_line(s):
    """
    :param s: the text segment
    :return: the cleaned text segment
    """
    # clean
    s = s.replace(u'\xa0', ' ')
    s = contractions.fix(s)
    s = s.replace('\'', ' \' ')
    cleaned_line = s.translate(
        str.maketrans('', '', punctuation.replace(',-./:;', '').replace('@', '').replace('+', '').replace('\'',
                                                                                                          '') + '←' + '↑'))
    return cleaned_line.strip()

```
## 2
In the following code, you assume that a directory will be present.
- How do you ensure that users of your code know that this directory is needed?
```python
MODEL_DIR = 'sentence_classifier/models/run_2021_02_03_18_15_40_72271c125cfe'
```
## 3
Using the following code, you are not able to store different configurations of invalid pos tags.
This is useful when experimenting, so can keep a history of what worked and what not.
- How would you refactor for this scenario?
```python
INVALID_POS_TAGS = ['DET', 'PUNCT', 'ADP', 'CCONJ', 'SYM', 'NUM', 'PRON', 'SCONJ', 'ADV']
```
## 4
There are better ways to do do the following.
- Please refactor, so people who just run the script know what goes wrong if they do not fill out an argument
```python
data_dir = sys.argv[1]
```

## 5
Why will this behave differently on Windows and Linux?
```python
    text_as_list = page.split('\n')
```

