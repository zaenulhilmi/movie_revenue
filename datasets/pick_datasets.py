import pandas as pd

raw_dataset = pd.read_csv('./tmdb_5000_movies.csv')

use_dataset = raw_dataset[['revenue', 'title', 'popularity', 'budget', 'runtime', 'vote_average', 'vote_count']]

use_dataset.to_csv('./movies.csv', index=False)
