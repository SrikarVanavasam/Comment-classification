import re
import logging
import numpy as np
import pandas as pd
from collections import Counter

def label_comment (row):
    if row['severe_toxic'] == 1 :
        return 'severe_toxic'
    if row['toxic'] == 1 :
        return 'toxic'
    if row['threat'] == 1 :
        return 'threat'
    if row['obscene'] == 1:
        return 'obscene'
    if row['insult']  == 1:
        return 'insult'
    if row['identity_hate'] == 1:
        return 'identity_hate'
    return 'none'

def clean_str(s):
    """Clean sentence"""
    s = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", s)
    s = re.sub(r"\'s", " \'s", s)
    s = re.sub(r"\'ve", " \'ve", s)
    s = re.sub(r"n\'t", " n\'t", s)
    s = re.sub(r"\'re", " \'re", s)
    s = re.sub(r"\'d", " \'d", s)
    s = re.sub(r"\'ll", " \'ll", s)
    s = re.sub(r",", " , ", s)
    s = re.sub(r"!", " ! ", s)
    s = re.sub(r"\(", " \( ", s)
    s = re.sub(r"\)", " \) ", s)
    s = re.sub(r"\?", " \? ", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r'\S*(x{2,}|X{2,})\S*',"xxx", s)
    s = re.sub(r'[^\x00-\x7F]+', "", s)
    return s.strip().lower()

def load_data_and_labels(filename):
    """Load sentences and labels"""
    #df = pd.read_csv(filename, compression='zip', dtype={'comment_text': object})
    df = pd.read_csv(filename, engine='python')
    non_selected = ['severe_toxic','toxic','threat',	'obscene',	'insult', 'identity_hate']
    df['comment_type'] = df.apply (lambda row: label_comment (row),axis=1)
    
    #df = df.['comment_type', 'comment_text'].copy()
    selected = ['comment_type', 'comment_text']
    #non_selected = list(set(df.columns) - set(selected))
    
    df = df.drop(non_selected, axis=1) # Drop non selected columns
    df = df.dropna(axis=0, how='any', subset=selected) # Drop null rows
    df = df.reindex(np.random.permutation(df.index)) # Shuffle the dataframe
    
    # Map the actual labels to one hot labels
    labels = sorted(list(set(df[selected[0]].tolist())))
    one_hot = np.zeros((len(labels), len(labels)), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))
    
    x_raw = df[selected[1]].apply(lambda x: clean_str(x)).tolist()
    y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()
    print(df['comment_type'])
    return x_raw, y_raw, df, labels
    
def batch_iter(data, batch_size, num_epochs, shuffle=True):
	"""Iterate the data batch by batch"""
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int(data_size / batch_size) + 1

	for epoch in range(num_epochs):
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data

		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield shuffled_data[start_index:end_index]

    
if __name__ == '__main__':
    input_file = './data/train.csv'
    load_data_and_labels(input_file)
