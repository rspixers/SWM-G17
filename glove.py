import pandas as pd
from IPython.display import display
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

file_list = ['amazon5', 'amazon15', 'amazon30', 'amazon60', 'amazon240', 'amazon1440', 'apple5', 'apple15', 'apple30', 'apple60', 'apple240', 'apple1440']

for files in file_list:
    path = "data/"+files+".csv"
    file_name = pd.read_csv(path)

    # display(file_name)
    unnamed = []
    for i in file_name['Unnamed: 0']:
        unnamed.append(i)
    print(len(unnamed))

    # file_name.info()
    
    

    # load the Stanford GloVe model
    def get_glove_embeddings(df):
        word_list = []
        for i in df['filteredtext']:
            x = i[1:-1].split(", ")
            words = []
            for j in x:
                s = j.split(" ")
                for k in s:
                    words.append(k)
            word_list.append(words)
        filename = './data/glove.6B.100d.txt.word2vec'
        model = KeyedVectors.load_word2vec_format(filename, binary=False)


        embedding_list = []
        for i in word_list:
            embeddings = []
            for j in i:
                try:
                    glov = model[j]
                    embeddings.append(glov)
                except:
                    continue
            embedding_list.append(embeddings)
        return embedding_list
    
    print(len(get_glove_embeddings(file_name)))

    glove_dict = {}
    glove_dict['Unnamed: 0'] = unnamed
    glove_dict['Glove_embedding'] = get_glove_embeddings(file_name)

    glove_df = pd.DataFrame.from_dict(glove_dict)

    joined_df = file_name.merge(glove_df, how = 'inner', on = 'Unnamed: 0')
    store_path = "data/"+files+"_glove.csv"
    joined_df.to_csv(store_path, encoding="utf-8")


