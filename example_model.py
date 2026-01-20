import pandas as pd
import dill, os
import numpy as np
from sklearn.neighbors import NearestNeighbors as NN

# Machine Learning Algorithm: Definition and Training
class ML:
    def __init__(self,profile,condition,respon):
        self.respo   = respon
        self.profile = profile
        self.knn = NN(n_neighbors=condition.shape[0]).fit(condition) #KNeighborsTimeSeries(n_neighbors=condition.shape[0]).fit(condition)

    def predict(self,features):
        from tslearn.neighbors import KNeighborsTimeSeries as KNTS
        prediction = {}
        for i in features.keys():
            array = features[i]["Initial_Conditions"]["initial_status"].values.reshape(1,51)
            dist, ind = self.knn.kneighbors(array, return_distance=True)
            best_cand = []
            good_cand = []
            for j in range(len(dist[0])):
                if dist[0][j] == 0:
                    best_cand.append(ind[0][j])
                else:
                    good_cand.append(ind[0][j])
            if len(best_cand) > 0:
                # Online Train 
                array_p = self.profile[best_cand,:,:]
                ts_knn = KNTS(n_neighbors=1).fit(array_p)
                # Predict
                test_x = features[i]["Profiles"].values.reshape(1,72,3)
                d_ind = ts_knn.kneighbors(test_x, return_distance=False)
                final_ind = best_cand[d_ind[0][0]]
            else:
                # Online Train 
                array_p = self.profile[good_cand,:,:]
                ts_knn = KNTS(n_neighbors=1).fit(array_p)
                # Predict
                test_x = features[i]["Profiles"].values.reshape(1,72,3)
                d_ind = ts_knn.kneighbors(test_x, return_distance=False)
                final_ind = good_cand[d_ind[0][0]]
            prediction[i] = self.respo[final_ind]
        return prediction

# save the model into a dill file
def save_class(model):
    # Save the model in a library
    with open('model.dill', 'wb') as file:
        dill.dump(model,file)

def main():  

    # Getting Input Data
    pro_e = []
    ini_e = []
    respo_v = {}
    count = 0
    for instance in os.listdir("Train_Data"):
        ex_in = pd.read_excel("Train_Data/"+instance+"/explanatory_variables.xlsx",sheet_name=None,index_col=0)
        pro_e.append(ex_in["Profiles"].values.tolist())
        ini_e.append(ex_in["Initial_Conditions"]["initial_status"].tolist())
        respo_v[count] = pd.read_excel("Train_Data/"+instance+"/Response_Variables.xlsx",sheet_name="is_on",index_col=0)
        count+=1
    pro_array = np.array(pro_e)
    ini_array = np.array(ini_e)

    # Cretate and save the class 
    save_class(ML(pro_array,ini_array,respo_v))

if __name__ == "__main__":
    main()
