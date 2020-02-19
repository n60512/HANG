from nltk.corpus import stopwords
import nltk
# nltk.download('stopwords')
word_list = 'the feel and texture of the wallet is like the skin of a it s too slippery at all and you won t like it that much if you are sensitive to cheap plastic type leather also there s hardly any room for condoms'
filtered_words = [word for word in word_list.split(' ') if word not in stopwords.words('english')]

print(word_list)
print()
print(filtered_words)