### Boris Zarubin
### b.zarubin@innopolis.university
### B21-DS-01
# MovieLens recommendation system
A recommender system is a type of information filtering system that suggests items or content
to users based on their interests, preferences, or past behavior. These systems are commonly
used in various domains, such as e-commerce, entertainment, social media, and online content
platforms.

## Installation
Install [Tensorflow](https://www.tensorflow.org/install).
```shell
git clone https://github.com/Poseidondon/movies-recommender
pip install -r requirements.txt
```

## Training
*You can skip this part if you want to use pre-trained model.*

Follow [2.0-training-and-visualization.ipynb](notebooks/2.0-training-and-visualization.ipynb) to
train a model and save weights.

## Prediction
Python
```python
from benchmark.movie_recommender import MovieRecommender

model_path = 'models/best/DCN.ckpt'
movie_rec = MovieRecommender(model_path)
# here you select user for which you wish to receive a recommendation
user_id = 345
print(movie_rec.predict(user_id, recommend_seen=False))
```

## Evaluation
CLI
```shell
python benchamrk/evaluate.py

RMSE:                   0.9815
Factorized top 10:      0.016300
Factorized top 50:      0.148750
Factorized top 100:     0.310900
```
Also, you could run [4.0-recmetrics](notebooks/4.0-recmterics.ipynb) to get MAP@K and MAR@K scores:
```shell
MAP@1:	0.006369426751592357
MAR@1:	0.0012060627506487378
MAP@5:	0.006340233545647558
MAR@5:	0.0034913429306858796
MAP@10:	0.0077628714161610225
MAR@10:	0.00620951802509859
```
