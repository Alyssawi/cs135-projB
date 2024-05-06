import pandas as pd
from surprise import Dataset, NormalPredictor, Reader, SVD, accuracy
from surprise.model_selection import cross_validate
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import sklearn.model_selection
import sklearn.pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np


from train_valid_test_loader import load_train_valid_test_datasets

# Load the dataset in the same way as the main problem 
train_tuple, valid_tuple, test_tuple, n_users, n_items = \
        load_train_valid_test_datasets()

user_info = pd.read_csv('../data_movie_lens_100k/user_info.csv', usecols=[1,2])
movie_info = pd.read_csv('../data_movie_lens_100k/movie_info.csv', usecols=[2])
print(user_info)
print(movie_info)


def tuple_to_surprise_dataset(tupl):
    """
    This function convert a subset in the tuple form to a `surprise` dataset. 
    """
    ratings_dict = {
        "userID": tupl[0],
        "itemID": tupl[1],
        "rating": tupl[2],
    }

    df = pd.DataFrame(ratings_dict)

    # A reader is still needed but only the rating_scale param is requiered.
    reader = Reader(rating_scale=(1, 5))

    # The columns must correspond to user id, item id and ratings (in that order).
    dataset = Dataset.load_from_df(df[["userID", "itemID", "rating"]], reader)

    return dataset


## Below we train an SVD model and get its vectors 

# train an SVD model using the training set
trainset = tuple_to_surprise_dataset(train_tuple).build_full_trainset()
# print(trainset)
algo = SVD(n_factors=10)
algo.fit(trainset)


# Use an example to show to to slice out user and item vectors learned by the SVD 
uid = valid_tuple[0][0]
iid = valid_tuple[1][0]
rui = valid_tuple[2][0]

# Get model parameters
# NOTE: the SVD model has its own index system because the storage using raw user and item ids
# is not efficient. We need to convert raw ids to internal ids. Please read the few lines below
# carefully 

mu = algo.trainset.global_mean # SVD does not even fit mu -- it directly use the rating mean 
bu = algo.bu[trainset.to_inner_uid(uid)]
bi = algo.bi[trainset.to_inner_iid(iid)] 
pu = algo.pu[trainset.to_inner_uid(uid)] 
qi = algo.qi[trainset.to_inner_iid(iid)]

print(f"pu shape: {algo.pu.shape}, qi shape: {algo.qi.shape}")
# print(trainset.to_inner_iid(1674))



def build_good_table(tupl):
    # dataset = tuple_to_surprise_dataset(tuple)
    
    concatenatedcrap = np.zeros((tupl[0].size, (algo.n_factors * 2 + 3)))
    print(concatenatedcrap.shape)
    print(tupl[0].shape)



    for i in range(concatenatedcrap.shape[0]):
        uid = tupl[0][i]
        iid = tupl[1][i]
        # print(f"building for user {uid} and item {iid}\n")
        try:
            pu = algo.pu[trainset.to_inner_uid(uid)] 
        except:
            pu = np.zeros(algo.n_factors)
        try:
            qi = algo.qi[trainset.to_inner_iid(iid)]
        except:
            qi = np.zeros(algo.n_factors)
        ui = user_info.iloc[uid].values
        vj = movie_info.iloc[iid].values
        # test = pd.concat([ui, vj])
        
        # pu_df = pd.DataFrame(pu.reshape(-1, algo.n_factors), columns=[f'pu_{x}' for x in range(len(pu))])
        # qi_df = pd.DataFrame(qi.reshape(-1, algo.n_factors), columns=[f'qi_{x}' for x in range(len(qi))])
        # pdlist = [pu_df, qi_df, ui.T.reset_index(i), vj.T.reset_index(i)]
        concatenatedcrap[i] = np.concatenate([pu, qi, ui, vj])

    return concatenatedcrap

features = build_good_table(train_tuple)

selector = SelectKBest(chi2, k=2)
features = selector.fit_transform(np.abs(features), (train_tuple[2] > 4.5))

lr_model = LogisticRegression(max_iter=1000, solver='liblinear')



param_grid = {'C': np.logspace(-9, 6, 15)}

lr_searcher = sklearn.model_selection.GridSearchCV(lr_model,
                param_grid=param_grid, scoring='roc_auc',
                return_train_score=True, refit=False, cv=10, verbose=3)
lr_searcher.fit(features, (train_tuple[2] > 4.5))
best_pipe = lr_model
best_pipe.set_params(**lr_searcher.best_params_)
best_pipe.fit(features, (train_tuple[2] > 4.5))

leaderboard = pd.read_csv('../data_movie_lens_100k/ratings_masked_leaderboard_set.csv')
lead_tuple = (
        leaderboard['user_id'].values,
        leaderboard['item_id'].values,
        leaderboard['rating'].values)

testset = build_good_table(lead_tuple)

output = best_pipe.predict(testset)

np.savetxt('predicted_ratings_leaderboardLR.txt', output, fmt='%i')
print(lr_searcher.best_params_)