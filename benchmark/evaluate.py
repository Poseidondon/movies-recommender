from pathlib import Path

from movie_recommender import MovieRecommender


if __name__ == '__main__':
    model_dir = Path(__file__).parent.parent / 'models'
    model_path = str(model_dir / 'best' / 'DCN.ckpt')
    movie_rec = MovieRecommender(model_path)
    tf_res = movie_rec.evaluate()
    print(f"RMSE:\t\t\t{tf_res['root_mean_squared_error']:.4f}")
    print(f"Factorized top 10:\t{tf_res['factorized_top_k/top_10_categorical_accuracy']:.6f}")
    print(f"Factorized top 50:\t{tf_res['factorized_top_k/top_50_categorical_accuracy']:.6f}")
    print(f"Factorized top 100:\t{tf_res['factorized_top_k/top_100_categorical_accuracy']:.6f}")
