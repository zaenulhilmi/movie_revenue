import pandas as pd

raw_dataset = pd.read_csv('./tmdb_5000_movies.csv')

df = raw_dataset[['revenue', 'popularity', 'budget', 'vote_average', 'vote_count']]

have_revenue = df['revenue'] > 0

df[have_revenue].to_csv('./movies.csv', index=False)
