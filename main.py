import numpy as np
import pandas as pd

from flask import Flask, render_template, request

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import json
import bs4 as bs
import urllib.request
import urllib.error
import pickle
import requests

from datetime import date, datetime


# ============================================================
# Load the NLP model and vectorizer from disk
# ============================================================

filename = 'nlp_model.pkl'

clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('tranform.pkl', 'rb'))


# ============================================================
# Convert string representation of list to Python list
# Example:
# '["abc","def"]' -> ['abc', 'def']
# ============================================================

def convert_to_list(my_list):

    my_list = my_list.split('","')

    if len(my_list) > 0:
        my_list[0] = my_list[0].replace('["', '')
        my_list[-1] = my_list[-1].replace('"]', '')

    return my_list


# ============================================================
# Convert string representation of number list to Python list
# Example:
# '[1,2,3]' -> ['1', '2', '3']
# ============================================================

def convert_to_list_num(my_list):

    my_list = my_list.split(',')

    if len(my_list) > 0:
        my_list[0] = my_list[0].replace('[', '')
        my_list[-1] = my_list[-1].replace(']', '')

    return my_list


# ============================================================
# Get movie suggestions for autocomplete
# ============================================================

def get_suggestions():

    data = pd.read_csv('main_data.csv')

    return list(data['movie_title'].str.capitalize())


# ============================================================
# Fetch IMDb reviews
# ============================================================

def get_imdb_reviews(imdb_id):

    reviews_list = []
    reviews_status = []

    if not imdb_id:
        return reviews_list, reviews_status

    try:

        # IMDb movie reviews URL
        url = f'https://www.imdb.com/title/{imdb_id}/reviews/'

        # Browser-like headers
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0 Safari/537.36'
            ),
            'Accept-Language': 'en-US,en;q=0.9'
        }

        req = urllib.request.Request(
            url,
            headers=headers
        )

        # Request IMDb
        sauce = urllib.request.urlopen(
            req,
            timeout=10
        ).read()

        # Parse HTML
        soup = bs.BeautifulSoup(
            sauce,
            'lxml'
        )

        # Find reviews
        soup_result = soup.find_all(
            'div',
            {
                'class': 'text show-more__control'
            }
        )

        # Analyze each review
        for review in soup_result:

            if review.string:

                review_text = review.string.strip()

                if not review_text:
                    continue

                reviews_list.append(review_text)

                # Send review to NLP model
                movie_review_list = np.array(
                    [review_text]
                )

                movie_vector = vectorizer.transform(
                    movie_review_list
                )

                pred = clf.predict(
                    movie_vector
                )

                # clf.predict() returns an array
                reviews_status.append(
                    'Positive'
                    if pred[0]
                    else 'Negative'
                )

    except urllib.error.HTTPError as e:

        # IMDb may block automated requests with 403
        print(
            f'IMDb HTTP Error: {e.code}'
        )

    except urllib.error.URLError as e:

        print(
            f'IMDb URL Error: {e.reason}'
        )

    except Exception as e:

        print(
            f'IMDb review processing error: {e}'
        )

    return reviews_list, reviews_status


# ============================================================
# Create Flask application
# ============================================================

app = Flask(__name__)


# ============================================================
# Home
# ============================================================

@app.route('/')
@app.route('/home')
def home():

    suggestions = get_suggestions()

    return render_template(
        'home.html',
        suggestions=suggestions
    )


# ============================================================
# Populate movie matches
# ============================================================

@app.route(
    '/populate-matches',
    methods=['POST']
)
def populate_matches():

    # Get data from AJAX request
    try:
        res = json.loads(
            request.get_data(
                as_text=True
            )
        )

    except Exception as e:

        print(
            f'Error reading AJAX data: {e}'
        )

        return 'Invalid request data', 400

    movies_list = res.get(
        'movies_list',
        []
    )

    # Create movie cards
    movie_cards = {}

    for movie in movies_list:

        poster_path = movie.get(
            'poster_path'
        )

        # Correct TMDB image URL
        poster_url = (
            'https://image.tmdb.org/t/p/original'
            + poster_path
            if poster_path
            else '/static/movie_placeholder.jpeg'
        )

        release_date = movie.get(
            'release_date',
            ''
        )

        if release_date:

            try:
                release_year = datetime.strptime(
                    release_date,
                    '%Y-%m-%d'
                ).year

            except ValueError:
                release_year = 'N/A'

        else:

            release_year = 'N/A'

        movie_cards[poster_url] = [
            movie.get('title', ''),
            movie.get('original_title', ''),
            movie.get('vote_average', ''),
            release_year,
            movie.get('id', '')
        ]

    return render_template(
        'recommend.html',
        movie_cards=movie_cards
    )


# ============================================================
# Recommend
# ============================================================

@app.route(
    '/recommend',
    methods=['POST']
)
def recommend():

    # ========================================================
    # Get data from AJAX/form request
    # ========================================================

    title = request.form.get(
        'title',
        ''
    )

    cast_ids = request.form.get(
        'cast_ids',
        ''
    )

    cast_names = request.form.get(
        'cast_names',
        ''
    )

    cast_chars = request.form.get(
        'cast_chars',
        ''
    )

    cast_bdays = request.form.get(
        'cast_bdays',
        ''
    )

    cast_bios = request.form.get(
        'cast_bios',
        ''
    )

    cast_places = request.form.get(
        'cast_places',
        ''
    )

    cast_profiles = request.form.get(
        'cast_profiles',
        ''
    )

    imdb_id = request.form.get(
        'imdb_id',
        ''
    )

    poster = request.form.get(
        'poster',
        ''
    )

    genres = request.form.get(
        'genres',
        ''
    )

    overview = request.form.get(
        'overview',
        ''
    )

    vote_average = request.form.get(
        'rating',
        ''
    )

    vote_count = request.form.get(
        'vote_count',
        ''
    )

    rel_date = request.form.get(
        'rel_date',
        ''
    )

    release_date = request.form.get(
        'release_date',
        ''
    )

    runtime = request.form.get(
        'runtime',
        ''
    )

    status = request.form.get(
        'status',
        ''
    )

    rec_movies = request.form.get(
        'rec_movies',
        ''
    )

    rec_posters = request.form.get(
        'rec_posters',
        ''
    )

    rec_movies_org = request.form.get(
        'rec_movies_org',
        ''
    )

    rec_year = request.form.get(
        'rec_year',
        ''
    )

    rec_vote = request.form.get(
        'rec_vote',
        ''
    )

    rec_ids = request.form.get(
        'rec_ids',
        ''
    )


    # ========================================================
    # Get movie suggestions for autocomplete
    # ========================================================

    suggestions = get_suggestions()


    # ========================================================
    # Convert strings to lists
    # ========================================================

    if rec_movies_org:
        rec_movies_org = convert_to_list(
            rec_movies_org
        )
    else:
        rec_movies_org = []

    if rec_movies:
        rec_movies = convert_to_list(
            rec_movies
        )
    else:
        rec_movies = []

    if rec_posters:
        rec_posters = convert_to_list(
            rec_posters
        )
    else:
        rec_posters = []

    if cast_names:
        cast_names = convert_to_list(
            cast_names
        )
    else:
        cast_names = []

    if cast_chars:
        cast_chars = convert_to_list(
            cast_chars
        )
    else:
        cast_chars = []

    if cast_profiles:
        cast_profiles = convert_to_list(
            cast_profiles
        )
    else:
        cast_profiles = []

    if cast_bdays:
        cast_bdays = convert_to_list(
            cast_bdays
        )
    else:
        cast_bdays = []

    if cast_bios:
        cast_bios = convert_to_list(
            cast_bios
        )
    else:
        cast_bios = []

    if cast_places:
        cast_places = convert_to_list(
            cast_places
        )
    else:
        cast_places = []


    # ========================================================
    # Convert number strings to lists
    # ========================================================

    if cast_ids:
        cast_ids = convert_to_list_num(
            cast_ids
        )
    else:
        cast_ids = []

    if rec_vote:
        rec_vote = convert_to_list_num(
            rec_vote
        )
    else:
        rec_vote = []

    if rec_year:
        rec_year = convert_to_list_num(
            rec_year
        )
    else:
        rec_year = []

    if rec_ids:
        rec_ids = convert_to_list_num(
            rec_ids
        )
    else:
        rec_ids = []


    # ========================================================
    # Clean review/character text
    # ========================================================

    for i in range(
        len(cast_bios)
    ):

        cast_bios[i] = (
            cast_bios[i]
            .replace(r'\n', '\n')
            .replace(r'\\"', '\\"')
        )


    for i in range(
        len(cast_chars)
    ):

        cast_chars[i] = (
            cast_chars[i]
            .replace(r'\n', '\n')
            .replace(r'\\"', '\\"')
        )


    # ========================================================
    # Create movie cards
    # ========================================================

    movie_cards = {}

    for i in range(
        min(
            len(rec_posters),
            len(rec_movies),
            len(rec_movies_org),
            len(rec_vote),
            len(rec_year),
            len(rec_ids)
        )
    ):

        movie_cards[
            rec_posters[i]
        ] = [
            rec_movies[i],
            rec_movies_org[i],
            rec_vote[i],
            rec_year[i],
            rec_ids[i]
        ]


    # ========================================================
    # Create cast dictionaries
    # ========================================================

    casts = {}

    for i in range(
        min(
            len(cast_names),
            len(cast_ids),
            len(cast_chars),
            len(cast_profiles)
        )
    ):

        casts[
            cast_names[i]
        ] = [
            cast_ids[i],
            cast_chars[i],
            cast_profiles[i]
        ]


    cast_details = {}

    for i in range(
        min(
            len(cast_names),
            len(cast_ids),
            len(cast_profiles),
            len(cast_bdays),
            len(cast_places),
            len(cast_bios)
        )
    ):

        cast_details[
            cast_names[i]
        ] = [
            cast_ids[i],
            cast_profiles[i],
            cast_bdays[i],
            cast_places[i],
            cast_bios[i]
        ]


    # ========================================================
    # Get IMDb reviews
    # ========================================================

    reviews_list = []
    reviews_status = []

    if imdb_id:

        reviews_list, reviews_status = get_imdb_reviews(
            imdb_id
        )


    # ========================================================
    # Get release date information
    # ========================================================

    movie_rel_date = ""
    curr_date = ""

    if rel_date:

        try:

            today = str(
                date.today()
            )

            curr_date = datetime.strptime(
                today,
                '%Y-%m-%d'
            )

            movie_rel_date = datetime.strptime(
                rel_date,
                '%Y-%m-%d'
            )

        except ValueError as e:

            print(
                f'Date processing error: {e}'
            )

            movie_rel_date = ""
            curr_date = ""


    # ========================================================
    # Combine reviews with sentiment
    # ========================================================

    movie_reviews = {
        reviews_list[i]: reviews_status[i]
        for i in range(
            min(
                len(reviews_list),
                len(reviews_status)
            )
        )
    }


    # ========================================================
    # Render recommendation page
    # ========================================================

    return render_template(
        'recommend.html',
        title=title,
        poster=poster,
        overview=overview,
        vote_average=vote_average,
        vote_count=vote_count,
        release_date=release_date,
        movie_rel_date=movie_rel_date,
        curr_date=curr_date,
        runtime=runtime,
        status=status,
        genres=genres,
        movie_cards=movie_cards,
        reviews=movie_reviews,
        casts=casts,
        cast_details=cast_details,
        suggestions=suggestions
    )


# ============================================================
# Run Flask application
# ============================================================

if __name__ == '__main__':

    app.run(
        debug=True
    )