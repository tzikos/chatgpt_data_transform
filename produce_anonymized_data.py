
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import pandas as pd
import hashlib

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def generate_hash(data):
    hash_object = hashlib.sha256()
    hash_object.update(data.encode())
    hash_hex = hash_object.hexdigest()

    return hash_hex

def clean_chats(text):
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    
    # POS tagging
    tagged_tokens = pos_tag(tokens)
    
    # Filter out verbs
    tokens = [word for word, pos in tagged_tokens if pos != 'VB' and word not in stop_words]
    
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

# Read and clean the dataset
df = pd.read_csv('chatgpt_data.csv')
df['clean_chat_text'] = df['chat_text'].apply(lambda x: clean_chats(x))

# Secure chat ids
id_df = pd.DataFrame(df['id'])
id_df['new_id']=id_df['id'].apply(lambda x: generate_hash(x))
df['new_id']=id_df['new_id']

# Base data
df['group'] = df['clean_chat_text'].apply(lambda x: 'Python' if 'python' in x else ('Tableau' if 'tableau' in x else ('SQL' if 'sql' in x else ('Excel' if 'excel' in x else ('Debugging' if 'error' in x else 'Other')))))
df['words_list']=df['clean_chat_text'].apply(lambda x: x.split(' '))
df_for_viz = df[['new_id','create_time','update_time','group']]

# Word count data
word_list = df[['new_id','words_list']].explode('words_list')
word_list.reset_index(drop=True,inplace=True)
word_count_df = word_list.groupby('new_id').value_counts().reset_index()

# Exports
id_df.to_csv('ids_match.csv',index=False,header=True)
df_for_viz.to_csv('base_chatgpt_data.csv',index=False,header=True,encoding='utf-8')
word_count_df.to_csv('word_count_data.csv',index=False,header=True,encoding='utf-8')