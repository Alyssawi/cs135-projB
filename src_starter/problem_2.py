
import pandas as pd
from surprise import Dataset, NormalPredictor, Reader, SVD, accuracy
from surprise.model_selection import cross_validate
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import sklearn.model_selection
import numpy as np


from train_valid_test_loader import load_train_valid_test_datasets

# Load the dataset in the same way as the main problem 
train_tuple, valid_tuple, test_tuple, n_users, n_items = \
        load_train_valid_test_datasets()

user_info = pd.read_csv('../data_movie_lens_100k/user_info.csv', usecols=[1,2,3])
movie_info = pd.read_csv('../data_movie_lens_100k/movie_info.csv', usecols=[2,3])
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
    
    concatenatedcrap = np.zeros((tupl[0].size, (algo.n_factors * 2 + 5)))
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
print(features)
print(features.shape)

base_gbdt = GradientBoostingClassifier(
    n_estimators=100,
    criterion='friedman_mse',
    max_depth=16,
    min_samples_split=2,
    min_samples_leaf=1)

gbdt_hyperparameter_grid_by_name = dict(
    max_features=[6, 12, 18],
    max_depth=[8, 12],
    learning_rate=[0.1, 0.2],
    n_estimators=[50, 100],
    random_state=[101],
    )


gbdt_searcher = sklearn.model_selection.GridSearchCV(base_gbdt,
                param_grid=gbdt_hyperparameter_grid_by_name, scoring='accuracy',
                return_train_score=True, refit=False, cv=3, verbose=3)

gbdt_searcher.fit(features, train_tuple[2])

best_gbdt = base_gbdt
best_gbdt.set_params(**gbdt_searcher.best_params_) # TODO call set_params using the best_params_ found by your searcher
best_gbdt.fit(features, train_tuple[2])

leaderboard = pd.read_csv('../data_movie_lens_100k/ratings_masked_leaderboard_set.csv')
lead_tuple = (
        leaderboard['user_id'].values,
        leaderboard['item_id'].values,
        leaderboard['rating'].values)

testset = build_good_table(lead_tuple)

output = best_gbdt.predict(testset)

np.savetxt('predicted_ratings_leaderboard.txt', output)

# Sanity check: we compute our own prediction and compare it against the model's prediction 
# our prediction
my_est = mu + bu + bi + np.dot(pu, qi) 

# the model's prediction
# NOTE: the training of the SVD model is random, so the prediction can be different with 
# different runs -- this is normal.   
svd_pred = algo.predict(uid, iid, r_ui=rui)

# The two predictions should be the same
print("My prediction: " + str(my_est) + ", SVD's prediction: " + str(svd_pred.est) + ", difference: " + str(np.abs(my_est - svd_pred.est)))

assert(np.abs(my_est - svd_pred.est) < 1e-6)


