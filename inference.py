import warnings
warnings.filterwarnings("ignore")
def prep_ing(x):
    return x.strip().replace(' ', '_')

import pandas as pd

tdf = pd.read_csv("archive/RecipeDB_Merged.csv")
class_list = list(tdf["Region"].unique())
# print(class_list)
clusters_df = {}
recipe_df = {}
for j in class_list:
    clusters_df[j] = []
    recipe_df[j]   = tdf[tdf["Region"] == j][["ingredients","Instructions"]]
    recipe_df[j]["ingredients"] = recipe_df[j].apply(lambda x: eval(x["ingredients"]),axis = 1)
    recipe_df[j]["ingredients"] = recipe_df[j].apply(lambda x: [prep_ing(i) for i in x["ingredients"]],axis=1)
    for i in range(1,6):
        clusters_df[j].append(pd.read_csv(f"archive(3)/f {j} {i}.csv"))
        clusters_df[j][-1]["len"] = clusters_df[j][-1].apply(lambda x : int(i), axis = 1)  

for i in clusters_df.keys():
    temp = pd.concat(clusters_df[i],axis=0)
    temp["tuple"] = temp.apply(lambda x: tuple([prep_ing(i) for i in eval(x["name"])]),axis = 1)
    clusters_df[i] = temp

    tup_map = {}
tup_ind = {}
itr = 0
for ind in clusters_df.keys():
    for _, row in clusters_df[ind].iterrows():
        if(row["tuple"] in tup_map.keys()):
            # print(row["tuple"] )
            continue
        tup_ind[itr] = row["tuple"]
        tup_map[row["tuple"]] = itr
        itr += 1
    clusters_df[ind]["name"] = clusters_df[ind].apply(lambda x: tup_map[x["tuple"]],axis = 1)
    clusters_df[ind] = clusters_df[ind][["name","len","tuple","ratio"]]


import torch
import numpy as np

with open("archive(4)/ingredient_embeddings.txt","r") as f:
    read_ings = f.read().split()

read_tensors = torch.load("archive(4)/final_ingredient_embeddings.pt")

raw_embedding_map = dict()

for i in range(len(read_ings)):
    raw_embedding_map[read_ings[i]] = read_tensors[read_ings[i]]

def get_tuple_embedding(tup_name: tuple[str], ing_emb):
    avg = None
    for i in tup_name:
        try:
            # print(i)
            avg += raw_embedding_map[i]
        except:
            avg = raw_embedding_map[i].clone()
            
    return avg/len(tup_name)

for i in clusters_df.keys():
    clusters_df[i]['tuple_embedding'] = clusters_df[i].apply(lambda x: get_tuple_embedding(x['tuple'],raw_embedding_map), axis = 1)

central_embeddings = {}
central_embeddings_ind = {}
for i in clusters_df.keys():
    central_embeddings[i] = (torch.mean(torch.stack(clusters_df[i]['tuple_embedding'].to_list()),axis=0))
    central_embeddings_ind[i] = (torch.mean(torch.stack(clusters_df[i][clusters_df[i]["len"] == 1]['tuple_embedding'].to_list()),axis=0))

def ___knn_predict(test_point,knn,top_k=1,fast_mode=True):
    distances = torch.cosine_similarity(knn["data"], test_point, dim=1)
    s, nearest_neighbors = torch.topk(distances, 1, largest=True)
    neighbor_labels = knn["label"][nearest_neighbors]
    predicted_label = torch.mode(neighbor_labels).values.item() 
    
    return predicted_label, s

def ___knn_predict2(test_point,knn,labels,top_k=1,fast_mode=True):
    distances = torch.cosine_similarity(knn, test_point, dim=1)
    s, nearest_neighbors = torch.topk(distances, top_k, largest=True)
    x =s * labels[nearest_neighbors][:,1]
    neighbor_labels = labels[nearest_neighbors][np.argmax(x),0]
    return int(neighbor_labels)

__knn_predict = ___knn_predict2

bti = pd.read_csv("ALUCRAD.CSV")

def get_tuples_in_list(ing_list: list[str], tuple_to_code: dict,src:str): # correct implementation
    # print(ing_list.__class__)
    ing_cache = set(ing_list)
    tup_list = {
        1: [],
        2: [],
        3: [],
        4: [],
        5: []
    }
    # print(src)
    for i in clusters_df[src]["tuple"]:
        temp = True
        for j in i:
            if j not in ing_cache:
                temp = False
                break
        if temp:
            tup_list[len(i)].append(tuple_to_code[i])
    return tup_list

def cluster_selection_stratergy1(tuple_list, cluster_level,src:str):
    ratio_list = []
    name_list  = []
    for i in tuple_list.keys():
        if i > cluster_level:
            break
        for j in tuple_list[i]:
            temp = clusters_df[src][clusters_df[src]["name"] == j]
            ratio_list.append(temp["ratio"].iloc[0])
            name_list.append(j)
    tr =  np.array(ratio_list)
    tn =  np.array(name_list)
    del ratio_list, name_list
    name_list = tn[np.argsort(tr)[::-1]]
    del tr, tn
    return name_list

def __tuple_decoder2(orignal_tuple, foreign_tuple,tup_map): # TODO: Might need better replacement protocol
    return (tup_map[orignal_tuple],tup_map[foreign_tuple])
tuple_decoder = __tuple_decoder2

def calc_threshold(ing_old,ing_new):
    if len(ing_old) < len(ing_new):
        ing_new, ing_old = ing_old,ing_new
    t = set(ing_new)
    thres = 0
    for i in ing_old:
        if i not in t:
            thres += 1
    return thres       


def __get_foriegn_embeddings2(name_org,mapping,dest):
    return int(mapping[dest][mapping["index"] == name_org].iloc[0])

def __translate_list2(old_ing_list,mapping):
    new_ing_list = [num for num in old_ing_list if num not in mapping[0]]
    new_ing_list.extend(list(mapping[1]))
    return new_ing_list

translate_list = __translate_list2
get_foriegn_embeddings = __get_foriegn_embeddings2

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
no_classes = 5
no_ingredients =len(read_ings)  # number of unique ings for 5 classes
hidden_layer = int(math.sqrt(no_classes*no_ingredients))

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(no_ingredients, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, no_classes)
        )
    def forward(self, x):
        '''Forward pass'''
        x=torch.squeeze(x).float()
        x=self.layers(x)
        return x

ing_dict= {}
for i in range(len(read_ings)):
    ing_dict[read_ings[i]] = i
def ret_vec_for_model(ing_list):
    a=np.zeros(len(ing_dict.keys()),dtype=float)
    for i in ing_list:
        a[ing_dict[i]] = 1
    a=torch.from_numpy(a)
    return a

model = MLP()
model = torch.load("ind_ita_vib_fin.pt",map_location=torch.device('cpu'))
class_label = {}
for i in class_list:
    class_label[i] = len(class_label)

def check_transformation_bert(model: MLP, old_ing: list,new_ing: list,label_old:str="",label_new:str=""): 
    # works
    model.eval()
    inp= torch.mean(torch.stack([raw_embedding_map[i] for i in old_ing]),axis=0)
    inp2= torch.mean(torch.stack([raw_embedding_map[i] for i in new_ing]),axis=0)
    pred=model(torch.stack([inp,inp2]))
    _, predicted = torch.max(pred.data, 1)
    if(predicted[0] == predicted[1]): # this might be wrong
        return False
    return True

def check_transformation(model: MLP, old_ing: list,new_ing: list,label_old:str="",label_new:str=""): 
    # works
    model.eval()
    inp= torch.tensor(ret_vec_for_model(old_ing))
    inp2= torch.tensor(ret_vec_for_model(new_ing))
    len(inp)
    pred=model(torch.stack([inp,inp2]))
    _, predicted = torch.max(pred.data, 1)
    if(predicted[1] == class_label[label_old]): # this might be wrong
        return False
    elif(predicted[1] != class_label[label_new]):
        return None
    return True

def __helper2(ingredient_list,name_list,src,dest,upper_bound=99999):
    thershold = 0
    success = False
    ing_old = ingredient_list
    cache = set()
    for i in range(len(name_list)):
        tester = tup_ind[name_list[i]]
        skip = False
        for j in tester:
            if(j in cache):
                skip = True
                break
            else:
                cache.add(j)
        if skip:
            continue
        italian_name = get_foriegn_embeddings(name_list[i], bti,dest)
        ing_new = translate_list(ing_old,tuple_decoder(int(name_list[i]),int(italian_name),tup_ind))
        if(check_transformation(model,ing_old,ing_new, src, dest)):
            success = True
            break
        else:
            ing_old = ing_new
    thershold = calc_threshold(ingredient_list,ing_new)
    if(not success):
        thershold = 1000
    return thershold, ing_new 
helper = __helper2

def get_threshold(ingredient_list, tuple_list,src,dest): #returns mininum number of substitution to transform recipe
    name_list = cluster_selection_stratergy1(tuple_list,5,src)
    threshold_new, ing_new_2 = helper(ingredient_list,name_list,src,dest)
    threshold = threshold_new
    return  threshold, ing_new_2

# infererence_df = pd.read_csv("inference.csv")
# infererence_df["ingredients"] = infererence_df.apply(lambda x: eval(x["ingredients"]),axis = 1)
# infererence_df["ingredients"] = infererence_df.apply(lambda x: [prep_ing(i) for i in x["ingredients"]],axis=1)

# answer_df = pd.DataFrame({"ingredients":[],"converted_ingredients":[],"src":[],"dest":[],"is_converted":[],"threshold":[]})
# for _,x in infererence_df[i].iterrows():
#         inger = x["ingredients"]
#         t = get_threshold(inger,get_tuples_in_list(inger,tup_map,i),i,j)
#         if(t[0] == 1000):

import argparse

def main():
    parser = argparse.ArgumentParser(description="Process command-line arguments with options.")

    # Define the options
    parser.add_argument("-s", "--source", required=True, help="Source string")
    parser.add_argument("-d", "--destination", required=True, help="Destination string")
    parser.add_argument("-i", "--items", required=True, help="Item list as a Python list (e.g., '[item1, item2, item3]')")

    # Parse the arguments
    args = parser.parse_args()

    # Convert items to a Python list
    try:
        items = eval(args.items)  # Assumes the user provides a valid Python list
        items = [prep_ing(i) for i in items]
        if not isinstance(items, list):
            raise ValueError
    except:
        parser.error("Item list must be a valid Python list (e.g., '[item1, item2, item3]')")
    if(args.source not in class_list):
        parser.error("Need source to be from " + str(class_list))
    if(args.destination not in class_list):
        parser.error("Need destination to be from " + str(class_list))

    # Process inputs
    print("Source String:", args.source)
    print("Destination String:", args.destination)
    print("Item List:", items)
    results = get_threshold(items,get_tuples_in_list(items,tup_map,args.source),args.source,args.destination)
    similar_item  = 0
    for i in results[1]:
        for j in items:
            if i == j:
                similar_item += 1
    print("Converted Item List: ",results[1])
    print("Is Converted: ", results[0] != 1000)
    print("No. of new and mordified ingredients added: ",results[0])
    print('No of unaffected items: ',similar_item)
    print("Orignal list length: ", len(items))
    print("Final list length: ", len(results[1]))


if __name__ == "__main__":
    main()