import requests, gzip, json, random, pickle
from transformers import DistilBertTokenizerFast

# URLs for Goodreads dataset
genre_url_dict = {
    'poetry': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz',
    'children': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz',
    'comics_graphic': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz',
    'fantasy_paranormal': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz',
    'history_biography': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz',
    'mystery_thriller_crime': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz',
    'romance': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz',
    'young_adult': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz'
}

def load_reviews(url, head=10000, sample_size=2000):
    reviews, count = [], 0
    response = requests.get(url, stream=True)
    with gzip.open(response.raw, 'rt', encoding='utf-8') as file:
        for line in file:
            d = json.loads(line)
            reviews.append(d['review_text'])
            count += 1
            if head is not None and count >= head:
                break
    return random.sample(reviews, min(sample_size, len(reviews)))

def prepare_data(model_name="distilbert-base-cased", max_length=512):
    genre_reviews_dict = {g: load_reviews(u, head=100000, sample_size=2000)
                          for g, u in genre_url_dict.items()}
    pickle.dump(genre_reviews_dict, open('genre_reviews_dict.pickle', 'wb'))

    train_texts, train_labels, test_texts, test_labels = [], [], [], []
    for genre, reviews in genre_reviews_dict.items():
        reviews = random.sample(reviews, 1000)
        for r in reviews[:800]:
            train_texts.append(r); train_labels.append(genre)
        for r in reviews[800:]:
            test_texts.append(r); test_labels.append(genre)

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    test_encodings  = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)

    return train_encodings, test_encodings, train_labels, test_labels

