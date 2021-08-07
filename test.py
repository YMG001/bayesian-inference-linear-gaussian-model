import auxiliary_functions as utils
import numpy as np


G = utils.G


 # Testing GetMu #
#################
data,labels,individuals = utils.load_dataset()
data_train = data[:,:,0:int(2045*0.8)]   #1636 of instances (80%) to train model
data_test = data[:,:,int(2045*0.8):2045] #409 of instances (20%) to evaluate model
labels_train = labels[0:int(2045*0.8)]
labels_test = labels[int(2045*0.8):2045]

M = utils.learn_model(data_train,labels_train,G)

mu = utils.GetMu(data[:,:,0],2, 0, 1, M)
print(mu)


 # Printing Naïve Bayes model #
##############################
data,labels,individuals = utils.load_dataset()
m = utils.learn_model(data,labels)

for i in range(0, 20):
    print(i)
#    for j in range(0, 3):
    print('Means:')
    print(m.jointparts[i].means[0])
    print('Sigmas:')
    print(m.jointparts[i].sigmas[0])


 # Testing learn_model #
#######################

data,labels,individuals = utils.load_dataset()
m = utils.learn_model(data,labels,G)

print('Special case of HIP_CENTER')
print('Means')
print(m.jointparts[0].HC_means[0][0,0]) #jointparts[ <<joint>> ].HC_means[0][<<x,y,z>>,<<class(0-3)>>]
print('Sigmas')
print(m.jointparts[0].HC_sigmas[0][0,0])
print()

print('Betas')
print(m.jointparts[1].betas[0]) #jointparts[ <<joint>> ].betas[0][<<x,y,z>>,<<class(0-3)>>,<<beta(0-3)>>]
print('Sigmas')
print(m.jointparts[1].sigmas[0])


 # Testing fit_linear_gaussian 1 #
#################################

betas = np.array([1,2,10,4,5,6])
sigma = 0.1
n = 100
X,Y = utils.generate_random_lgm_samples(n,betas,sigma)

# This following call should output  betas and sigma close to the above ones
b, s = utils.fit_linear_gaussian(Y,X)

print( 'GT_Betas: ' + str(betas))
print ('My_Betas: ' + str(b))
print('GT_Sigma: ' + str(sigma))
print ('My_Sigma: ' + str(s))


 # Testing fit_linear_gaussian 2 #
#################################

import scipy.io

dd = scipy.io.loadmat('data/ejemplolineargaussian.mat')
dd = dd['ejemplo'][0]
# The inputs are
X = dd['inputX']
Y = dd['inputY']
X = X[0]
Y = Y[0][:,0]
# The expected outputs are
betas = dd['outputBetas']
sigma = dd['outputSigma']

### Mine
data,labels,individuals = utils.load_dataset()
#Y_,X_ = getInstancesAndParentValues(data,labels,1,1,1,G)

b,s = utils.fit_linear_gaussian(Y,X)
print('GT_Betas:' + str(betas[0][:,0]))
print('My_Betas:' + str(b))

print('GT_Sigma: ' + str(sigma[0][:,0]))
print('My_Sigma: ' + str(s))



 # Testing fit_gaussian #
#######################

V = np.array([1,2,10,4,5,6])

mean, var = utils.fit_gaussian(V)

print(mean)
print(var)

 # NB Model data validation #
############################

dd = scipy.io.loadmat('data/validation_data.mat')
data = dd['data_small'] # Input data
labels = dd['labels_small'] # Input labels
individuals = dd['individuals_small'] # Input individual indexes
dd['train_indexes'] # Instances used for training
dd['test_indexes']  # Instances used for test
dd['model_nb']      # NB model
dd['model_lg']      # LG model
dd['accur_nb']      # Accuracy of NB model on test instances
dd['accur_lg']      # Accuracy of LG model on test instances

### Mine
data_ori,labels_ori,individuals_ori = utils.load_dataset()
m = utils.learn_model(data,labels)
with np.printoptions(precision=3, suppress=True):
    for i in range(0,20):
        print('## Joint '+ str(i) + ' ##')
        print('My_Means:')
        print(m.jointparts[i].means[0])
        print('GT_Means: ')
        print(str(dd['model_nb'][0,0][0][0][i][0]))
        print()

        print('My_Sigmas_:')
        print(m.jointparts[i].sigmas[0])
        print('GT_Sigmas: ')
        print(str(dd['model_nb'][0,0][0][0][i][1]))

        print()
        print()



 # LGM Model data validation #
#############################

dd = scipy.io.loadmat('data/validation_data.mat')
data = dd['data_small'] # Input data
labels = dd['labels_small'] # Input labels
individuals = dd['individuals_small'] # Input individual indexes
dd['train_indexes'] # Instances used for training
dd['test_indexes']  # Instances used for test
dd['model_nb']      # NB model
dd['model_lg']      # LG model
dd['accur_nb']      # Accuracy of NB model on test instances
dd['accur_lg']      # Accuracy of LG model on test instances

### Mine
data_ori,labels_ori,individuals_ori = utils.load_dataset()
m = utils.learn_model(data,labels,G)
with np.printoptions(precision=3, suppress=True):
        
    print('## Joint '+ str(0) + ' ##')
    print('My_HCMeans:')
    print(m.jointparts[0].HC_means[0])
    print('GT_HCMeans: ')
    print(str(dd['model_lg'][0,0][0][0][0][0]))
    print()

    print('My_HCSigmas:')
    print(m.jointparts[0].HC_sigmas[0])
    print('GT_HCSigmas: ')
    print(str(dd['model_lg'][0,0][0][0][0][1]))

    print()
    print()   
    
    for i in range(1,20):
        print('## Joint '+ str(i) + ' ##')
        print('My_Betas:')
        print(m.jointparts[i].betas[0])
        print('GT_Betas: ')
        print(str(dd['model_lg'][0,0][0][0][i][0]))
        print()

        print('My_Sigmas_:')
        print(m.jointparts[i].sigmas[0])
        print('GT_Sigmas: ')
        print(str(dd['model_lg'][0,0][0][0][i][1]))

        print()
        print()



