import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import tensorflow_recommenders as tfrs

from typing import Dict, Text


ratings = tfds.load("movielens/100k-ratings", split="train")
movies = tfds.load("movielens/100k-movies", split="train")

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
movies = movies.map(lambda x: x["movie_title"])

timestamps = np.concatenate(list(ratings.map(lambda x: x["timestamp"]).batch(100)))

max_timestamp = timestamps.max()
min_timestamp = timestamps.min()

# init data for embeddings
timestamp_buckets = np.linspace(min_timestamp, max_timestamp, num=1000,)

unique_movie_titles = np.unique(np.concatenate(list(movies.batch(1000))))
unique_user_ids = np.unique(np.concatenate(list(ratings.batch(1_000).map(
    lambda x: x["user_id"]))))
unique_occupations = np.unique(np.concatenate(list(ratings.batch(1_000).map(
    lambda x: x["user_occupation_text"]))))
unique_zipcodes = np.unique(np.concatenate(list(ratings.batch(1_000).map(
    lambda x: x["user_zip_code"]))))
unique_genders = np.unique(np.concatenate(list(ratings.batch(1_000).map(
    lambda x: x["user_gender"]))))
unique_ages = np.unique(np.concatenate(list(ratings.batch(1_000).map(
    lambda x: x["bucketized_user_age"]))))


class UserModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, int(len(unique_user_ids) ** 0.25)),
        ])
        self.timestamp_embedding = tf.keras.Sequential([
            tf.keras.layers.Discretization(timestamp_buckets.tolist()),
            tf.keras.layers.Embedding(len(timestamp_buckets) + 1, int(len(timestamp_buckets) ** 0.25)),
        ])
        self.normalized_timestamp = tf.keras.layers.Normalization(axis=None)
        self.normalized_timestamp.adapt(timestamps)

        self.occupation_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_occupations, mask_token=None),
            tf.keras.layers.Embedding(len(unique_occupations) + 1, int(len(unique_occupations) ** 0.25)),
        ])
        self.zipcode_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_zipcodes, mask_token=None),
            tf.keras.layers.Embedding(len(unique_zipcodes) + 1, int(len(unique_zipcodes) ** 0.25)),
        ])
        self.gender_embedding = tf.keras.Sequential([
            tf.keras.layers.IntegerLookup(vocabulary=unique_genders, mask_token=None),
            tf.keras.layers.Embedding(len(unique_genders) + 1, int(len(unique_genders) ** 0.25)),
        ])
        self.age_embedding = tf.keras.Sequential([
            tf.keras.layers.IntegerLookup(vocabulary=unique_ages, mask_token=None),
            tf.keras.layers.Embedding(len(unique_ages) + 1, int(len(unique_ages) ** 0.25)),
        ])

    def call(self, inputs):
        # Take the input dictionary, pass it through each input layer,
        # and concatenate the result.
        return tf.concat([
            self.user_embedding(inputs["user_id"]),
            self.timestamp_embedding(inputs["timestamp"]),
            tf.reshape(self.normalized_timestamp(inputs["timestamp"]), (-1, 1)),
            self.occupation_embedding(inputs["user_occupation_text"]),
            self.zipcode_embedding(inputs["user_zip_code"]),
            self.gender_embedding(inputs["user_gender"]),
            self.age_embedding(inputs["bucketized_user_age"]),
        ], axis=1)


class QueryModel(tf.keras.Model):
    """Model for encoding user queries."""

    def __init__(self, layer_sizes, projection_dim=None):
        """Model for encoding user queries.

        Args:
          layer_sizes:
            A list of integers where the i-th entry represents the number of units
            the i-th layer contains.
        """
        super().__init__()

        # We first use the user model for generating embeddings.
        self.embedding_model = UserModel()

        # Cross & Dense layer
        self._cross_layer = tfrs.layers.dcn.Cross(
            projection_dim=projection_dim,
            kernel_initializer="glorot_uniform")

        # Then construct the layers.
        self.dense_layers = tf.keras.Sequential()

        # Use the ReLU activation for all but the last layer.
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))

        # No activation for the last layer.
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size))

    def call(self, inputs):
        feature_embedding = self.embedding_model(inputs)
        feature_embedding = self._cross_layer(feature_embedding)
        return self.dense_layers(feature_embedding)


class MovieModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        max_tokens = 10_000

        self.title_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_movie_titles, mask_token=None),
            tf.keras.layers.Embedding(len(unique_movie_titles) + 1, 32)
        ])

        self.title_vectorizer = tf.keras.layers.TextVectorization(max_tokens=max_tokens)

        self.title_text_embedding = tf.keras.Sequential([
            self.title_vectorizer,
            tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
            tf.keras.layers.GlobalAveragePooling1D(),
        ])

        self.title_vectorizer.adapt(movies)

    def call(self, titles):
        return tf.concat([
            self.title_embedding(titles),
            self.title_text_embedding(titles),
        ], axis=1)


class CandidateModel(tf.keras.Model):
    """Model for encoding movies."""

    def __init__(self, layer_sizes, projection_dim=None):
        """Model for encoding movies.

        Args:
          layer_sizes:
            A list of integers where the i-th entry represents the number of units
            the i-th layer contains.
        """
        super().__init__()

        self.embedding_model = MovieModel()

        # Cross & Dense layer
        self._cross_layer = tfrs.layers.dcn.Cross(
            projection_dim=projection_dim,
            kernel_initializer="glorot_uniform")

        # Then construct the layers.
        self.dense_layers = tf.keras.Sequential()

        # Use the ReLU activation for all but the last layer.
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))

        # No activation for the last layer.
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size))

    def call(self, inputs):
        feature_embedding = self.embedding_model(inputs)
        feature_embedding = self._cross_layer(feature_embedding)
        return self.dense_layers(feature_embedding)


class MovielensModel(tfrs.models.Model):
    def __init__(self,
                 rating_weight: float,
                 query_layer_sizes,
                 candidate_layer_sizes,
                 projection_dim=None):
        assert query_layer_sizes[-1] == candidate_layer_sizes[
            -1], f'query and candidate embeddings of different sizes: {query_layer_sizes[-1]} and {candidate_layer_sizes[-1]}'

        super().__init__()
        self.query_model = QueryModel(query_layer_sizes, projection_dim=projection_dim)
        self.candidate_model = CandidateModel(candidate_layer_sizes, projection_dim=projection_dim)
        # A small model to take in user and movie embeddings and predict ratings.
        # We can make this as complicated as we want as long as we output a scalar
        # as our prediction.
        self.rating_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1),
        ])

        # The tasks.
        self.rating_task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )
        self.retrieval_task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=movies.batch(128).map(self.candidate_model),
            ),
        )

        # The loss weights.
        self.rating_weight = rating_weight
        self.retrieval_weight = 1 - rating_weight

    def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
        query_embeddings = self.query_model({
            "user_id": features["user_id"],
            "user_gender": features["user_gender"],
            "user_zip_code": features["user_zip_code"],
            "user_occupation_text": features["user_occupation_text"],
            "bucketized_user_age": features["bucketized_user_age"],
            "timestamp": features["timestamp"],
        })
        candidate_embeddings = self.candidate_model(features["movie_title"])

        x = tf.concat([query_embeddings, candidate_embeddings], axis=1)

        return (
            query_embeddings,
            candidate_embeddings,
            # We apply the multi-layered rating model to a concatentation of
            # user and movie embeddings.
            self.rating_model(x),
        )

    def compute_loss(self, features, training=False):
        ratings = features.pop("user_rating")

        query_embeddings, candidate_embeddings, rating_predictions = self(features)

        # We compute the loss for each task.
        rating_loss = self.rating_task(
            labels=ratings,
            predictions=rating_predictions,
        )
        retrieval_loss = self.retrieval_task(query_embeddings, candidate_embeddings)

        # And combine them using the loss weights.
        return (self.rating_weight * rating_loss + self.retrieval_weight * retrieval_loss)
