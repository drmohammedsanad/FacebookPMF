import numpy as np
import pandas
from numpy import linalg as LA
import cPickle as pickle
from sklearn import metrics
import math


class PMF:
    def __init__(self, num_feat=1, epsilon=1, _lambda=0.1, momentum=0.8, maxepoch=10, num_batches=10, batch_size=1000):
        self.num_feat = num_feat
        self.epsilon = epsilon
        self._lambda = _lambda
        self.momentum = momentum
        self.maxepoch = maxepoch
        self.num_batches = num_batches
        self.batch_size = batch_size
        
        self.w_C = None
        self.w_I = None

        self.err_train = []
        self.err_val = []
        
    def fit(self, train_vec, val_vec):   
        # mean subtraction
        self.mean_inv = np.mean(train_vec[:,2])
        
        pairs_tr = train_vec.shape[0]
        pairs_va = val_vec.shape[0]
        num_inv = 6426
        num_com = 8638

        incremental = False
        if ((not incremental) or (self.w_C is None)):
            # initialize
            self.epoch = 0
            self.w_C = 0.1 * np.random.randn(num_com, self.num_feat)
            self.w_I = 0.1 * np.random.randn(num_inv, self.num_feat)
            
            self.w_C_inc = np.zeros((num_com, self.num_feat))
            self.w_I_inc = np.zeros((num_inv, self.num_feat))
        
        
        while self.epoch < self.maxepoch:
            self.epoch += 1

            # Shuffle training truples
            shuffled_order = np.arange(train_vec.shape[0])
            np.random.shuffle(shuffled_order)

            # Batch update
            for batch in range(self.num_batches):
                # print "epoch %d batch %d" % (self.epoch, batch+1)

                batch_idx = np.mod(np.arange(self.batch_size * batch,
                                             self.batch_size * (batch+1)),
                                   shuffled_order.shape[0])

                batch_invID = np.array(train_vec[shuffled_order[batch_idx], 0], dtype='int32')
                batch_comID = np.array(train_vec[shuffled_order[batch_idx], 1], dtype='int32')

                # Compute Objective Function
                pred_out = np.sum(np.multiply(self.w_I[batch_invID,:], 
                                                self.w_C[batch_comID,:]),
                                axis=1) # mean_inv subtracted

                rawErr = pred_out - train_vec[shuffled_order[batch_idx], 2] + self.mean_inv

                # Compute gradients
                Ix_C = 2 * np.multiply(rawErr[:, np.newaxis], self.w_I[batch_invID,:]) \
                        + self._lambda * self.w_C[batch_comID,:]
                Ix_I = 2 * np.multiply(rawErr[:, np.newaxis], self.w_C[batch_comID,:]) \
                        + self._lambda * self.w_I[batch_invID,:]
            
                dw_C = np.zeros((num_com, self.num_feat))
                dw_I = np.zeros((num_inv, self.num_feat))

                # loop to aggreate the gradients of the same element
                for i in range(self.batch_size):
                    dw_C[batch_comID[i],:] += Ix_C[i,:]
                    dw_I[batch_invID[i],:] += Ix_I[i,:]


                # Update with momentum
                self.w_C_inc = self.momentum * self.w_C_inc + self.epsilon * dw_C / self.batch_size
                self.w_I_inc = self.momentum * self.w_I_inc + self.epsilon * dw_I / self.batch_size


                self.w_C = self.w_C - self.w_C_inc
                self.w_I = self.w_I - self.w_I_inc

                # Compute Objective Function after
                if batch == self.num_batches - 1:
                    pred_out = np.sum(np.multiply(self.w_I[np.array(train_vec[:,0], dtype='int32'),:],
                                                    self.w_C[np.array(train_vec[:,1], dtype='int32'),:]),
                                        axis=1) # mean_inv subtracted
                    rawErr = pred_out - train_vec[:, 2] + self.mean_inv
                    obj = LA.norm(rawErr) ** 2 \
                            + 0.5*self._lambda*(LA.norm(self.w_I) ** 2 + LA.norm(self.w_C) ** 2)
                    print(np.sqrt(obj/pairs_tr))

                    self.err_train.append(np.sqrt(obj/pairs_tr))

                # Compute validation error
                if batch == self.num_batches - 1:
                    pred_out = np.sum(np.multiply(self.w_I[np.array(val_vec[:,0], dtype='int32'),:],
                                                    self.w_C[np.array(val_vec[:,1], dtype='int32'),:]),
                                        axis=1) # mean_inv subtracted
                    rawErr = pred_out - val_vec[:, 2] + self.mean_inv
                    self.err_val.append(LA.norm(rawErr)/np.sqrt(pairs_va))

    def predict(self, invID): 
        return np.dot(self.w_C, self.w_I[invID,:]) + self.mean_inv
        
    def set_params(self, parameters):
        if isinstance(parameters, dict):
            self.num_feat = parameters.get("num_feat", 10)
            self.epsilon = parameters.get("epsilon", 1)
            self._lambda = parameters.get("_lambda", 0.1)
            self.momentum = parameters.get("momentum", 0.8)
            self.maxepoch = parameters.get("maxepoch", 10)
            self.num_batches = parameters.get("num_batches", 10)
            self.batch_size = parameters.get("batch_size", 1000)
def rmse(predictions, actual):
    count = 0
    differences_squared = 0
    for i in range(len(actual)):
        for j in range(len(actual[i])):
            if actual[i][j] > 0:
                differences = predictions[i][j] - actual[i][j]  # the DIFFERENCEs.
                differences_squared = differences_squared + differences ** 2  # the SQUAREs of ^
                count = count + 1
    mean_of_differences_squared = differences_squared / float(count)  # the MEAN of ^
    rmse_val = np.sqrt(mean_of_differences_squared)  # ROOT of ^
    return rmse_val


def topn(R, n, u):  # Calculate the list of n recommended items
    rr = np.array(R[u])
    sorted = np.argsort(R[u])[::-1]
    rr1 = rr[sorted]
    print len(sorted[0:n])

    print(rr1[0:n])

    s = []
    top_n = sorted[0:n]
    return top_n, rr1[0:n]

if __name__ == "__main__":

    train_list_original = [i.strip().split(",") for i in
                           open('../Data/matrix_fatorization_data_indexed_3_normalized_Train.csv', 'r').readlines()]
    train_df_original = pandas.DataFrame(train_list_original, columns=['uid', 'fid', 'interactions'], dtype=float)
    R_norm_train1 = train_df_original.pivot(index='uid', columns='fid', values='interactions').fillna(0)
    R_norm_train = R_norm_train1.as_matrix()

    test_list_original = [i.strip().split(",") for i in
                          open('../Data/matrix_fatorization_data_indexed_3_normalized_Test.csv', 'r').readlines()]
    test_df_original = pandas.DataFrame(test_list_original, columns=['uid', 'fid', 'interactions'], dtype=float)
    R_norm_test1 = test_df_original.pivot(index='uid', columns='fid', values='interactions').fillna(0)
    R_norm_test = R_norm_test1.as_matrix()
    
    eval = []
    eval1 = []
    t = True
    for K in range(1, 6, 1):
        rmse1_E = 0
        rmse2_E = 0
        for E in range(1, 11, 1):

            print('###############', K)
            print('EEEEEEEEEEEEEEE',E)

            '''
            instance1 = PMF(num_feat=K)
            instance1.fit(R_norm_train, R_norm_test)
            P = instance1.w_I
            Q = instance1.w_C
            #pickle.dump(P, open("../Experiments/Facebook_PMF1/P/" + str(E) + "_P60k_k" + str(K) + "_first_90train", "wb"))
            #pickle.dump(Q, open("../Experiments/Facebook_PMF1/Q/" + str(E) + "_Q60k_k" + str(K) + "_first_90train", "wb"))
            '''
            nP = pickle.load(open("../Experiments/Facebook_PMF1/P/" + str(E) + "_P60k_k" + str(K) + "_first_90train", "rb"))
            nQ = pickle.load(open("../Experiments/Facebook_PMF1/Q/" + str(E) + "_Q60k_k" + str(K) + "_first_90train", "rb"))
            
            nR = np.dot(nP, nQ.T) 
            
            rmse1 = rmse(nR, R_norm_train)
            rmse2 = rmse(nR, R_norm_test)            
            
            arr = []
            All_data = [i.strip().split(",") for i in open('../Data/matrix_fatorization_data_indexed_ALL.csv','r').readlines()]
            All_data1 = pandas.DataFrame(All_data, columns=['M_uid','A_uid', 'M_fid','A_fid', 'interactions'], dtype=float)

            No_of_friends = [i.strip().split(",") for i in open('../Data/matrix_fatorization_data_indexed_No_of_Friends.csv','r').readlines()]
            No_of_friends1 = pandas.DataFrame(No_of_friends, columns=['M_uid', 'interactions'], dtype=int)

            h = [100, 1404, 2208, 3534, 4791]
            m = [82, 83, 93, 230, 271, 1972]
            l = [814, 898, 979, 1096, 1904, 2253, 2354, 2518]
            test = [100]
            for uHigh1 in l:
                for i in range(len(All_data1['M_uid'])):
                    if All_data1['M_uid'][i] == uHigh1:
                        #print'I found it'
                        found = All_data1['A_uid'][i]

                for i in range(len(No_of_friends1['M_uid'])):
                    if No_of_friends1['M_uid'][i] == uHigh1:
                        found1 = No_of_friends1['interactions'][i]
                print('Top 10 for '+str(uHigh1))
                arrID, arrVal = topn(nR, 10, uHigh1-1)
                arrAID = []
                for r in arrID:
                    for j in range(len(All_data1)):
                        if r == All_data1['M_fid'][j]:
                            arrAID.append(All_data1['A_fid'][j])
                            break
                        elif r == All_data1['M_uid'][j]:
                            arrAID.append(All_data1['A_uid'][j])
                            break
                arr1 = []
                t = True
                for j in range(len(arrID)):
                    if t:
                        arr.append(str(uHigh1) + "," + str(found) + "," + str(found1) + "," + str(arrID[j]) + "," + str(arrAID[j]) + "," + str(arrVal[j]) + "," + "%.2f" % float(arrVal[j]*84))
                        t = False
                    else:
                        arr.append('' + "," + '' + "," + '' + "," + str(arrID[j]) + "," + str(arrAID[j]) + "," + str(arrVal[j]) + "," + "%.2f" % float(arrVal[j]*84))
            evaluation = pandas.DataFrame(arr, columns=['eval1'])
            evaluation.to_csv('../Experiments/Facebook_PMF1/Results/updated/AnalysisLow.csv', sep='\n', header=False, float_format='%.2f', index=False, )


            rmse1_E = rmse1_E + rmse1
            rmse2_E = rmse2_E + rmse2

            print('RMSE Train: ', "%.4f" % rmse1, "%.4f" % rmse1_E)
            print('RMSE Test: ', "%.4f" % rmse2, "%.4f" % rmse2_E)

        rmse1m = rmse1_E / E
        rmse2m = rmse2_E / E
        if t:
            eval.append(
                "Metrics" + "," + "K" + "," + "Train RMSE" + "," + "Test RMSE")
            t = False
        eval.append("PMF" + "," + str(K) + "," + "%.4f" % rmse1m + "," + "%.4f" % rmse2m)

    evaluation = pandas.DataFrame(eval1, columns=['eval'])
    evaluation.to_csv('../Experiments/Facebook_PMF/Results/Facebook_PMF_Evaluation_first90train.csv', sep='\n', header=False, float_format='%.2f', index=False, )