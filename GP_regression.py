import numpy as np
import logging
import random
from scipy.spatial.transform import Rotation as R

class scalar_para_von_mises_RBF_kernel:
    def __init__(self, feature_num, default_theta = 2e-2):
        self. theta = default_theta
    def compute( self, input1, input2):
         Rot_1 = R.from_euler('zyx', input1[0:3], degrees=False)
         Rot_2 = R.from_euler('zyx', input2[0:3], degrees=False)
         First_part = self. theta * np.exp( np.trace( Rot_1.as_matrix().T * Rot_2.as_matrix()) )
         Second_part = self. theta * np.exp(-   pow( np.linalg.norm((input1[3:] - input2[3:])), 2) )
         return (First_part + Second_part)

class scalar_para_von_mises_RBF_kernel_with_9_rot_vec:
    def __init__(self, feature_num, default_theta_Rot = 0.8, default_theta_Vec = 0.8, default_l = 0.9):
        self. sigma1 = default_theta_Rot
        self. sigma2 = default_theta_Vec
        self. l      = default_l
    def compute( self, input1, input2):
        # print('shape of input:', np.shape(input1))
        # if input1.ndim == 1:
        #     fst_input1 = input1[0:9]
        # else:
        #  exp_in = np.dot(input1[0:9],input2[0:9])
        #  First_part = self. theta1 * np.exp( exp_in )
        #  Second_part = self. theta2 * np.exp(-   np.linalg.norm((input1[9:] - input2[9:])**2) )
        #  return (First_part + Second_part)


        # print('shape_of_input1:', np.shape(input1))
        # print('shape_of_input2:', np.shape(input2))

        fst_input1 = input1[:,0:9]
        fst_input2 = input2[:,0:9]

        # print('shape_of_fst_input1:', np.shape(fst_input1))
        # print('shape_of_fst_input2:', np.shape(fst_input2))

        mul_matrix = np.matmul(fst_input1, fst_input2.T)
        # print('mul_matrix:',mul_matrix)


        fst_mat = self. sigma1  ** 2 * np.exp( mul_matrix)

        # print('fst_mat:', fst_mat)

        sec_input1 = input1[:,9:]
        sec_input2 = input2[:,9:]

        # print('np.sum(sec_input1**2, 1).reshape(-1, 1):', np.sum(sec_input1**2, 1).reshape(-1, 1))

        dist_matrix = np.sum(sec_input1**2, 1).reshape(-1, 1) + np.sum(sec_input2**2, 1) - 2 * np.dot(sec_input1, sec_input2.T)

        # print('dist_matrix:', dist_matrix)

        sec_mat = self. sigma2  ** 2 * np.exp(- 0.5 / self. l ** 2 * dist_matrix)
        # print('sec_mat:', sec_mat)

        return  np.multiply(fst_mat, sec_mat)


class square_exponential_kernel:
    def __init__(self, feature_num, default_sigma_f = 0.8, default_l = 0.8):
        self. sigma = default_sigma_f
        self. l     = default_l
    def compute( self, input1, input2):
         dist_matrix = np.sum(input1**2, 1).reshape(-1, 1) + np.sum(input2**2, 1) - 2 * np.dot(input1, input2.T)
         return  self. sigma  ** 2 * np.exp(- 0.5 / self. l ** 2 * dist_matrix)

class Gaussian_Process_Regression:
        def __init__(self, kernel_name):
            self. kernel_class    = globals()[kernel_name]
            self. has_fitted = False

        def cache(self, train_X, train_Y):
            self. X = train_X
            self. Y = train_Y
            [Xraw, Xcol] = np.shape(self. X)
            [Yraw, Ycol] = np.shape(self. Y)
            self. output_dim = Ycol

            if Xraw != Yraw:
                logging.error("Data dimensions of input and output do not equal!!!")
            self. feature_num = Xraw
            self. feature_dim = Xcol
            self. kernel = self. kernel_class(self.feature_num )

        def fit(self, train_X, train_Y):
            self. X = train_X
            self. Y = train_Y
            # print('self. X', self. X)
            [Xraw, Xcol] = np.shape(self. X)
            [Yraw, Ycol] = np.shape(self. Y)
            self. output_dim = Ycol

            if Xraw != Yraw:
                logging.error("Data dimensions of input and output do not equal!!!")
            self. feature_num = Xraw
            self. feature_dim = Xcol

            self. kernel = self. kernel_class(self.feature_num )

            self. Gram_matrix =  self. kernel. compute(self. X, self. X)
            self. Gram_inv =np.linalg.pinv(np.mat( self. Gram_matrix) + 1e-10 * np.eye(self. Gram_matrix.shape[0]))
            print('shape_of_Gram_inv:', np.shape(self. Gram_inv))
            self. has_fitted = True

        
        def remove_feature_at(self, i ):
             self. X = np.delete(self. X, i, 0)
             self. Y = np.delete(self. Y, i, 0)
             self.fit(self. X, self. Y)
            
        def predict(self, input_X):
            # [Xraw, Xcol] = np.shape(input_X)
            # if Xcol != self. feature_dim:
            #       logging.error("Prediction input dimensions of input do not equal to training!!!")
            # print('Xraw',Xraw)
            # print('self. output_dim',self. output_dim)

            Kzz = self.kernel. compute(input_X, input_X)
            KXz = self.kernel. compute(self.X, input_X)


            mu_predict = KXz.T.dot(self. Gram_inv).dot(self.Y)
            cov_predict = Kzz - KXz.T.dot(self. Gram_inv).dot(KXz)

            cov_predict[cov_predict<0] = 0
            return (mu_predict, cov_predict)

                 # [feature_num] *  [feature_num X feature_num] * [feature_num X output_dim] = [1 X output_dim]

def ConEn_optimize(gpr:Gaussian_Process_Regression, target_num, initial_num, batch_size):
    serials = random.sample(range(gpr. feature_num),initial_num)
    X_data = gpr.X[serials,:]
    Y_data = gpr.Y[serials,:]

    remain_data_serial = set([i for i in range(gpr.feature_num)]) - set(serials)

    count = target_num - initial_num

    while count > 0:
        Gram_matrix= gpr.kernel.compute(X_data, X_data)
        Gram_inv =np.linalg.pinv(np.mat( Gram_matrix) + 1e-10 * np.eye(Gram_matrix.shape[0]))

        list_remain_data_serial = list(remain_data_serial)
        
        critics_index = [0.0] * len(list_remain_data_serial)

        # for h in range(len (list_remain_data_serial)):
        #     # X_data_temp = np.row_stack((X_data, gpr.X[list_remain_data_serial[h]]))
        #     # Gram_matrix_after = gpr.kernel.compute(X_data_temp, X_data_temp)
        #     # print('gpr.X[list_remain_data_serial[h]]:', gpr.X[list_remain_data_serial[h]])
        #     # print('gpr.X[list_remain_data_serial[h]]:', np.mat(gpr.X[list_remain_data_serial[h]]))
        Kzz = gpr. kernel. compute(gpr.X[list_remain_data_serial,:],gpr.X[list_remain_data_serial,:])
        print('shape_of_Kzz:', np.shape(Kzz))
        KXz = gpr. kernel. compute(X_data, gpr.X[list_remain_data_serial,:])
        cov_predict = Kzz - KXz.T.dot( Gram_inv).dot(KXz)

        critics_index =  1.96 *  np.array(np.sqrt(np.diag(cov_predict)))

        if count > batch_size:
            md_batch_size = batch_size
        else:
            md_batch_size = count

        critic_ind_sort =  np.argsort(critics_index)
        choose_sort = critic_ind_sort[(len(critics_index) - md_batch_size) : len(critics_index)]
        # choose_sort = critic_ind_sort[0 :md_batch_size]
        print('choose_sort:',choose_sort)

        selected_data_serial = [ list_remain_data_serial[i] for i in choose_sort]
        X_data = np.row_stack((X_data, gpr.X[selected_data_serial,:]))
        Y_data = np.row_stack((Y_data, gpr.Y[selected_data_serial,:]))

        remain_data_serial = remain_data_serial - set(selected_data_serial)
       
        
        count = count - md_batch_size

        print('ConEn_optimize process', 100 * (1 - (count) / (target_num - initial_num)), '%')
        
        # if count <= 0:
        #     break;

    gpr.fit(X_data, Y_data)

    return gpr

            # Y_data_temp = np.row_stack((Y_data, gpr.Y[list_remain_data_serial[h]]))

def DGram_optimize(gpr:Gaussian_Process_Regression, target_num, compare_seed_num):
    critics_index = np.zeros(gpr. feature_num) 
    for k in range(gpr. feature_num):
        sample_serial = set(range(gpr. feature_num))
        sample_serial = sample_serial - set([k])
        ran = random.sample(list(sample_serial),compare_seed_num)
        
        X_test = gpr.X[ran,:]
        X_test_plus_here = np.row_stack((X_test, np.array([gpr.X[k,:]])))
        Gram_matrix = gpr.kernel.compute(X_test_plus_here,X_test_plus_here)

        
        critics_index[k] = np.linalg.det(Gram_matrix)
        

        print('Gram_optimize process', 100 * (k+1) / gpr. feature_num, '%')

    critic_ind_sort =  np.argsort(critics_index)
    # print('critic_ind_sort',critic_ind_sort)
    # print('len(critics_index) - target_num',len(critics_index) - target_num)
    # print('len(critics_index)',len(critics_index))
    choose_sort = critic_ind_sort[(len(critics_index) - target_num) : len(critics_index)]
    # choose_sort = critic_ind_sort[0 :md_batch_size]
    print('choose_sort:',choose_sort)

    X_data = gpr.X[choose_sort,:]
    Y_data = gpr.Y[choose_sort,:]
    gpr.fit(X_data, Y_data)

    return gpr

def Error_optimize(gpr:Gaussian_Process_Regression, target_num, initial_num, batch_size):
    from scipy.stats import norm
    serials = random.sample(range(gpr. feature_num),initial_num)
    X_data = gpr.X[serials,:]
    Y_data = gpr.Y[serials,:]


    remain_data_serial = set([i for i in range(gpr.feature_num)]) - set(serials)
    # exp_critics = np.zeros(target_num - initial_num)
    count = target_num - initial_num
    
    while count > 0:
        Gram_matrix = gpr.kernel.compute(X_data, X_data)
        Gram_inv =np.linalg.pinv(np.mat(Gram_matrix) + 1e-10 * np.eye(Gram_matrix.shape[0]))
        print('Gram_inv_shape:', np.shape(Gram_inv))
        list_remain_data_serial = list(remain_data_serial)
            
        critics_index = [0.0] * len(list_remain_data_serial)

        # Kzz = gpr. kernel. compute( gpr.X[list_remain_data_serial],  gpr.X[list_remain_data_serial])
        KXz = gpr. kernel. compute( X_data,  gpr.X[list_remain_data_serial])

        # [Xraw, Xcol] = np.shape(gpr.X[list_remain_data_serial])
        mu_predict = KXz.T.dot(Gram_inv).dot(Y_data)

        # print('mu_predict:', mu_predict)
        # print('mu_predict_shape:', np.shape(mu_predict))
        # print('mu_predict_norm:', np.linalg.norm(mu_predict,axis = 1))
        critics_index = np.linalg.norm(mu_predict - \
                                       gpr.Y[list_remain_data_serial], axis = 1)

        if count > batch_size:
            md_batch_size = batch_size
        else:
            md_batch_size = count

        critic_ind_sort =  np.argsort(critics_index)
        choose_sort = critic_ind_sort[(len(critics_index) - md_batch_size) : len(critics_index)]
        # choose_sort = critic_ind_sort[0 :md_batch_size]
        print('choose_sort:',choose_sort)

        selected_data_serial = [ list_remain_data_serial[i] for i in choose_sort]
        print('list_remain_data_serial[choose_sort]:',selected_data_serial)
        X_data = np.row_stack((X_data, gpr.X[selected_data_serial,:]))
        Y_data = np.row_stack((Y_data, gpr.Y[selected_data_serial,:]))

        remain_data_serial = remain_data_serial - set(selected_data_serial)
    
        
        count = count - md_batch_size

        print('Error_optimize process', 100 * (1 - (count) / (target_num - initial_num)), '%')
            
    
    gpr.fit(X_data, Y_data)

    return gpr

def UCB_optimize(gpr:Gaussian_Process_Regression, target_num, initial_num, batch_size, sqr_beta = 10):
    from scipy.stats import norm
    serials = random.sample(range(gpr. feature_num),initial_num)
    X_data = gpr.X[serials,:]
    Y_data = gpr.Y[serials,:]


    remain_data_serial = set([i for i in range(gpr.feature_num)]) - set(serials)
    # exp_critics = np.zeros(target_num - initial_num)
    count = target_num - initial_num
    
    while count > 0:
        Gram_matrix = gpr.kernel.compute(X_data, X_data)
        Gram_inv =np.linalg.pinv(np.mat(Gram_matrix) + 1e-10 * np.eye(Gram_matrix.shape[0]))
        print('Gram_inv_shape:', np.shape(Gram_inv))
        list_remain_data_serial = list(remain_data_serial)
            
        # critics_index = [0.0] * len(list_remain_data_serial)

        # Kzz = gpr. kernel. compute( gpr.X[list_remain_data_serial],  gpr.X[list_remain_data_serial])
        Kzz = gpr. kernel. compute(gpr.X[list_remain_data_serial,:],gpr.X[list_remain_data_serial,:])
        print('shape_of_Kzz:', np.shape(Kzz))
        # KXz = gpr. kernel. compute(X_data, gpr.X[list_remain_data_serial,:])
        KXz = gpr. kernel. compute( X_data,  gpr.X[list_remain_data_serial])

        # [Xraw, Xcol] = np.shape(gpr.X[list_remain_data_serial])
        mu_predict = KXz.T.dot(Gram_inv).dot(Y_data)
        cov_predict = Kzz - KXz.T.dot( Gram_inv).dot(KXz)
        
        critics_index1 =  1.96 *  np.array(np.sqrt(np.diag(cov_predict)))
        critics_index2 = np.linalg.norm(mu_predict - \
                                       gpr.Y[list_remain_data_serial], axis = 1)       

        # print('mu_predict:', mu_predict)
        # print('mu_predict_shape:', np.shape(mu_predict))
        # print('mu_predict_norm:', np.linalg.norm(mu_predict,axis = 1))
        critics_index = critics_index1 + sqr_beta * critics_index2

        if count > batch_size:
            md_batch_size = batch_size
        else:
            md_batch_size = count

        critic_ind_sort =  np.argsort(critics_index)
        choose_sort = critic_ind_sort[(len(critics_index) - md_batch_size) : len(critics_index)]
        # choose_sort = critic_ind_sort[0 :md_batch_size]
        print('choose_sort:',choose_sort)

        selected_data_serial = [ list_remain_data_serial[i] for i in choose_sort]
        print('list_remain_data_serial[choose_sort]:',selected_data_serial)
        X_data = np.row_stack((X_data, gpr.X[selected_data_serial,:]))
        Y_data = np.row_stack((Y_data, gpr.Y[selected_data_serial,:]))

        remain_data_serial = remain_data_serial - set(selected_data_serial)
    
        
        count = count - md_batch_size

        print('UCB_optimize process', 100 * (1 - (count) / (target_num - initial_num)), '%')
            
    
    gpr.fit(X_data, Y_data)

    return gpr
