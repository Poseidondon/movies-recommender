import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_datasets as tfds
import numpy as np

from typing import Union
from pathlib import Path

try:
    from DCN import MovielensModel
except ModuleNotFoundError:
    from benchmark.DCN import MovielensModel


class MovieRecommender:
    def __init__(self,
                 cp_path: Union[str, Path],
                 movies: tf.data.Dataset = None):
        """
        Loads dataset and model int memory.

        :param cp_path: load model weights from that checkpoint
        :param movies: dataset of all movies
        """

        self.model = MovielensModel(0.5, [96, 32], [64, 32])
        self.model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
        self.model.load_weights(cp_path)

        ratings = tfds.load("movielens/100k-ratings", split="train")
        ratings = ratings.map(lambda x: {
            "movie_title": x["movie_title"],
            "user_id": x["user_id"],
            "user_rating": x["user_rating"],
            "user_gender": int(x["user_gender"]),
            "user_zip_code": x["user_zip_code"],
            "user_occupation_text": x["user_occupation_text"],
            "bucketized_user_age": int(x["bucketized_user_age"]),
            "timestamp": x["timestamp"],
        })
        self.ratings = ratings.batch(4096).cache()
        # train/test split is preserved due to random seed
        tf.random.set_seed(42)
        shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)
        self.test = shuffled.skip(80_000).take(20_000).batch(4096).cache()

        if movies is None:
            self.movies = tfds.load("movielens/100k-movies", split="train")
            self.movies = self.movies.map(lambda x: x["movie_title"])
        else:
            self.movies = movies

        # index used in prediction
        self.index = tfrs.layers.factorized_top_k.BruteForce(self.model.query_model)
        self.index.index_from_dataset(
            tf.data.Dataset.zip((self.movies.batch(100), self.movies.batch(100).map(self.model.candidate_model)))
        )

        # unbatched ratings
        self.unbatched_ratings = self.ratings.unbatch().cache()

    def evaluate(self):
        """
        Return tensorflow metrics on validation dataset (20% partition of movielens-1ooK)

        :return:
        """

        return self.model.evaluate(self.test, return_dict=True)

    def get_user_data(self, user_id):
        """
        Return user features used in DCN model in a convenient format

        :param user_id: user_id in MovieLens
        :return: tuple(user_data, movies_seen)
        """

        user_id = str(user_id)
        user_ratings = self.unbatched_ratings.filter(lambda x: x['user_id'] == user_id)
        user_data = list(user_ratings.take(1))[0]
        user_data = {
            "user_id": np.array([user_data["user_id"].numpy()]),
            "user_gender": np.array([user_data["user_gender"].numpy()]),
            "user_zip_code": np.array([user_data["user_zip_code"].numpy()]),
            "user_occupation_text": np.array([user_data["user_occupation_text"].numpy()]),
            "bucketized_user_age": np.array([user_data["bucketized_user_age"].numpy()]),
            "timestamp": np.array([user_data["timestamp"].numpy()])
        }
        movies_seen = np.unique(np.concatenate(list(user_ratings.batch(1_000).map(
            lambda x: x["movie_title"]))))[np.newaxis, ...]

        return user_data, movies_seen

    def predict(self, user_id, recommend_seen=False, k=10):
        """
        Predict movies by user_id

        :param user_id: user_id in MovieLens
        :param recommend_seen: if True - recommend movies that user already seen
        :param k: number of movies to recommend
        :return: tuple(scores, movie_titles)
        """

        user_data, movies_seen = self.get_user_data(user_id)
        if recommend_seen:
            scores, titles = self.index(user_data, k=k)
        else:
            scores, titles = self.index.query_with_exclusions(user_data, movies_seen, k=k)

        return scores.numpy()[0], titles.numpy()[0]


if __name__ == "__main__":
    # example of prediction
    model_dir = Path(__file__).parent.parent / 'models'
    model_path = str(model_dir / 'best' / 'DCN.ckpt')
    movie_rec = MovieRecommender(model_path)
    print(movie_rec.evaluate())
    print(movie_rec.predict("13"))
    print(movie_rec.predict("15"))
    print(movie_rec.predict("345"))
