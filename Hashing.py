import args


# create N_gram vector given a word
def n_gram(word, n=args.N):
    s = []
    word = '#' + word + '#'
    for j in range(len(word) - 2):
        s.append(word[j:j + 3])  # letter 3_gram
    return s


# create N_gram vector given a sentence
def lst_gram(lst, n=args.N):
    slice = []
    for word in str(lst).lower().split():
        slice.extend(n_gram(word))
    return slice


vocab = []  # initiate a vector to store letter n_grams
file_path = '/Users/wesley/Desktop/4579_final/MRPC/'  # data set path
files = ['train_data.csv', 'test_data.csv']  # training data & testing data set

for file in files:
    f = open(file_path + file, encoding='utf-8').readlines()
    for i in range(1, len(f)):  # line 0 is header
        s1, s2 = f[i][2:].strip('\n').split('\t')
        # delete the first 3 character of each line, tab space, and line change space, parse into two sentences
        vocab.extend(lst_gram(s1))
        vocab.extend(lst_gram(s2))

vocab = set(vocab) # clean list to be a non_duplicate set
vocab_list = ['[PAD]', '[UNK]']
vocab_list.extend(list(vocab))

# store the list in a separate file
vocab_file = args.VOCAB_FILE
with open(vocab_file, 'w', encoding='utf-8') as f:
    for sl in vocab_list:
        f.write(sl)
        f.write('\n')
