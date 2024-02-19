
from tqdm import tqdm
import networkx as nx
import pandas as pd
import pickle
import numpy as np
import torch
from sklearn.model_selection import train_test_split

def countSimilarities(data):
    similarity_count = {}

    for similarities in data.values():
        for _, similarity in similarities:
            if similarity not in similarity_count:
                similarity_count[similarity] = 1
            else:
                similarity_count[similarity] += 1

    return similarity_count
def create_user_item_dicts(rating_matrix):
    """
    convert rating matrix to user item dictionaries
    :param rating_matrix: rating matrix file, Pandas DataFrame [user_id, item_id, rating]
    :return: a tuple containing user and item rating dict.
            {userid1:{itemid1,itemid2,...}}
            {itemid1:{userid1,userid2,...}}
    """
    user_items_dict = {}
    item_users_dict={}

    for _, row in rating_matrix.iterrows():
        user_id = row['userid']
        item_id = row['itemid']
        rating = row['rating']

        if rating == 1:
            if user_id not in user_items_dict:
                user_items_dict[user_id] = {item_id}
            else:
                user_items_dict[user_id].add(item_id)

            if item_id not in item_users_dict:
                item_users_dict[item_id] = {user_id}
            else:
                item_users_dict[item_id].add(user_id)

    return user_items_dict, item_users_dict

def getSimForDict(set1, set2):
    """
    computer similarity between two sets
    :param set1:
    :param set2:
    :return: similarity
    """
    return len( set1 & set2 )

def getNeighbors( dataset, k ):
    sims = {}
    print("Start computering neighobrs for user or item......")
    for e1 in tqdm( dataset ):
        ulist=[]
        for e2 in dataset:
            if e1 == e2 :
                continue
            sim = getSimForDict( dataset[e1], dataset[e2] )
            if sim !=0:
                ulist.append((e2, sim))
        sims[e1] = [(i[0], i[1]) for i in sorted(ulist, key=lambda x: x[1], reverse=True) if i[1] != 0][:k]

    return sims

def build_graphs( train_data,val_data,test_data,k):
    """
    Build graphs from dataset include user-item graph, user relation graph, item relation graph
    :param train_data: train rating matrix file, Pandas DataFrame [user_id, item_id, rating]
    :param test_data: test rating matrix file, Pandas DataFrame [user_id, item_id, rating]
    :param k: number of neighbors
    :return: user_set, item_set, user_item graph, user relation graph, item relation graph
    """
    user_items_dict, item_users_dict = create_user_item_dicts(train_data)
    user_neighbors_dict=getNeighbors(user_items_dict,k)
    item_neighbors_dict=getNeighbors(item_users_dict,k)

    user_set_train = set(train_data['userid'].tolist())
    item_set_train = set(train_data['itemid'].tolist())
    user_set_val = set(val_data['userid'].tolist())
    item_set_val = set(val_data['itemid'].tolist())
    user_set_test = set(test_data['userid'].tolist())
    item_set_test = set(test_data['itemid'].tolist())
    user_set = user_set_train |user_set_val | user_set_test
    item_set = item_set_train | item_set_val | item_set_test

    num_user = max(user_set)+1
    num_item = max(item_set)+1
    user_item_set=user_set | {x+num_user for x in item_set}

    user_item_graph = nx.Graph()
    user_item_graph.add_nodes_from(user_item_set)

    user_graph = nx.Graph()
    user_graph.add_nodes_from(user_set)

    item_graph = nx.Graph()
    item_graph.add_nodes_from(item_set)

    for _, row in train_data.iterrows():
        user_id = row['userid']
        item_id = row['itemid']+num_user
        user_item_graph.add_edge(user_id, item_id)

    for user,nb_u in user_neighbors_dict.items():
        for u,sim in nb_u:
            user_graph.add_edge(user,u,weight=sim)

    for item,nb_i in item_neighbors_dict.items():
        for i,sim in nb_i:
            item_graph.add_edge(item,i,weight=sim)

    return user_set,item_set,user_item_graph,user_graph,item_graph

def get_user_item_set( rating_train,rating_test):
    """
    get user set and item set from train and test set
    :param rating_train: train rating matrix file, Pandas DataFrame [user_id, item_id, rating]
    :param rating_test: test rating matrix file, Pandas DataFrame [user_id, item_id, rating]
    :return: user set and item set
    """
    user_set_train = set(rating_train['userid'].tolist())
    item_set_train = set(rating_train['itemid'].tolist())
    user_set_test = set(rating_test['userid'].tolist())
    item_set_test = set(rating_test['itemid'].tolist())
    user_set=user_set_train | user_set_test
    item_set=item_set_train | item_set_test

    return user_set,item_set

def load_data(dataset,k,validation_prop):
    """
    load data from files
    :param dataset:
    :return:
    """
    # whole train set
    file_path_train = "datasets/"+dataset + '/train_ratings_b.pkl'
    train_ratings = pickle.load(open(file_path_train, 'rb'))
    train_ratings = pd.DataFrame(train_ratings, columns=['userid', 'itemid', 'rating'])

    # split train set and validation set
    train_data, val_data = train_test_split(train_ratings, test_size=validation_prop, random_state=42)
    print("Train set samples:", len(train_data))
    print("Validation set samples:", len(val_data))

    # test set
    file_path_test ="datasets/"+ dataset + '/test_ratings_b.pkl'
    test_data = pickle.load(open(file_path_test, 'rb'))
    test_data = pd.DataFrame(test_data, columns=['userid', 'itemid', 'rating'])

    user_set, item_set,user_item_graph, user_graph, item_graph = build_graphs(train_data,val_data, test_data,k)
    train_data, val_data,test_data = train_data.values.tolist(), val_data.values.tolist(),test_data.values.tolist()
    return train_data, val_data,test_data,user_set, item_set, user_item_graph, user_graph, item_graph

def graphSampling( G, nodes, n_size = 5, n_deep = 2 ):
    '''
    :param G: networkx
    :param nodes:
    :param n_size:
    :param n_deep:
    :return: torch.tensor
    '''
    leftEdges = [ ]
    rightEdges = [ ]

    for _ in range( n_deep ):
        target_nodes = list( set ( nodes ) )
        nodes = set( )
        for i in target_nodes:
            neighbors = list( G.neighbors( i ) )
            if len(neighbors) >= n_size:
                neighbors = np.random.choice( neighbors, size=n_size, replace = False )
            rightEdges.extend( neighbors )
            leftEdges.extend( [ i for _ in neighbors ] )
            nodes |= set( neighbors )
    edges = torch.tensor( [ leftEdges, rightEdges ], dtype = torch.long )
    return edges


if __name__ == '__main__':
    dataset="ml-1M"
    train_data, val_data, user_item_graph, user_graph, item_graph, train_data,test_data,user_set, item_set=load_data(dataset,10)
    print("Train set samples:", len(train_data))
    print("Validation set samples:", len(val_data))
    print("Test set samples:", len(test_data))
    print("count of user_set:", len(user_set))
    print("count of item_set:", len(item_set))
    print("user_set:",user_set)
    print("item_set:",item_set)

