import re
from nltk.util import ngrams
from collections import Counter


# Example text
s = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vestibulum aliquet dapibus justo, id euismod elit dictum at. Ut scelerisque luctus gravida. Vivamus a interdum leo. Praesent sit amet commodo urna. Quisque vitae tortor molestie, lacinia lorem vitae, egestas tortor. Aenean lacinia vestibulum ultrices. Vestibulum varius cursus bibendum. Praesent non tincidunt leo. Curabitur eu laoreet metus. Fusce turpis nunc, congue in ultrices quis, iaculis ut ante.

Integer molestie semper maximus. Cras ullamcorper pulvinar dictum. Integer rhoncus molestie metus quis pharetra. Phasellus cursus purus sed nulla accumsan, eu cursus nulla aliquet. Nam enim nunc, hendrerit vel aliquam in, interdum sit amet ligula."""
# Convert to lowercase, and only use words
s = s.lower()
s = re.sub(r"[^a-zA-Z0-9\s]", " ", s)
# Convert string to list of strings (i.e. split reduced sentence into words)
tokens = [token for token in s.split(" ") if token != ""]
# set n-value and change
n = 2
# get list of ngrams, JSON-formated
output = list(ngrams(tokens, n))
# convert to unique list of lists with count for each gram
output = [" ".join(gram) for gram in output]
output = [[gram, output.count(gram)] for gram in set(output)]
# output is the list of lists, [[gram, count],...])
print(output)
