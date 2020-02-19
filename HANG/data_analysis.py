#%%
from utils.DBconnector import DBConnection
from utils.preprocessing import Preprocess
from nltk.corpus import stopwords
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, mean, std
#%%
# Generating sentences that remove stopwords by nltk.
def generate_rm_sw_sentences(conn, interaction):
    
    pre_work = Preprocess()
    sql = "SELECT `ID`, reviewText FROM clothing_interaction{};".format(interaction)
    res = conn.selection(sql)

    for index in tqdm.tqdm(range(len(res))):
        
        id = res[index]['ID']
        reviewText = res[index]['reviewText']

        nor_sentence = pre_work.normalizeString(reviewText)
        
        sentence_segment = nor_sentence.split(' ')
        filtered_sentence = [word for word in sentence_segment if word not in stopwords.words('english')]

        filtered_sentence = " ".join(filtered_sentence)

        insert_sql = "INSERT INTO clothing_interaction{}_rm_sw (`ID`, reviewText) VALUES ('{}', '{}');".format(interaction ,id, filtered_sentence)
        conn.Insertion(insert_sql)  


# Generating sentences length 
def generate_sentences_length(conn, interaction=6):

    sql = "SELECT `ID`, reviewText FROM clothing_interaction{}_rm_sw;".format(interaction)

    res = conn.selection(sql)

    for index in tqdm.tqdm(range(len(res))):
        
        id = res[index]['ID']
        reviewText = res[index]['reviewText']
        
        sentence_segment = reviewText.split(' ')
        sentence_length = len(sentence_segment)
            
        insert_sql = "INSERT INTO clothing_interaction{}_sen_len (`ID`, sentence_length) VALUES ('{}', {});".format(interaction, id, sentence_length)
        conn.Insertion(insert_sql)


def Distribution(conn, interaction):
    
    sql = """
    select clothing_interaction{}_rm_sw.ID, clothing_interaction{}_sen_len.sentence_length 
    from clothing_interaction{}_rm_sw, clothing_interaction{}_sen_len 
    where clothing_interaction{}_rm_sw.ID = clothing_interaction{}_sen_len.ID
    ORDER BY sentence_length;
    """.format(interaction, interaction, interaction, interaction, interaction, interaction)
    res = conn.selection(sql)

    return res
    

def DrawDistibution(sentences_length):
    
    # print(sentences_length)

    
    plt.figure(figsize=(12, 8), dpi=144)
    plt.xlim([-10, 200])

    # matplotlib histogram
    # plt.hist(sentences_length, color = 'blue', edgecolor = 'black', bins=500)               

    x = np.random.gamma(6, size=200)
    sns.distplot(sentences_length, 
     bins=500)

    # Add labels
    plt.title('Frequency Histogram')
    plt.xlabel('Sentence Length')
    plt.ylabel('Frequency')

    plt.show()


def search_covering_area(sentences_length, percentage=90):


    pass

#%%
conn = DBConnection()

# generate_rm_sw_sentences(conn, 10)
# generate_sentences_length(conn, 10)

res = Distribution(conn, 6)
sentences_length = [row['sentence_length'] for row in res]

sen_mean = mean(sentences_length)
sen_std = std(sentences_length)

print(sen_mean
, sen_std)

interval = stats.norm.interval(0.90, sen_mean, sen_std)
print(interval)

# Draw in matplotlib
# DrawDistibution(sentences_length)

conn.close()