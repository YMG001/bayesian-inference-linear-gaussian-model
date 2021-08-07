import numpy as np 
import math
import random
import scipy.io
from copy import copy, deepcopy
from scipy.stats import lognorm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold


# Skeleton definition
NUI_SKELETON_POSITION_COUNT = 20

NONE = -1
HIP_CENTER = 0
SPINE = 1
SHOULDER_CENTER = 2
HEAD = 3
SHOULDER_LEFT = 4
ELBOW_LEFT = 5
WRIST_LEFT = 6
HAND_LEFT = 7
SHOULDER_RIGHT = 8
ELBOW_RIGHT = 9
WRIST_RIGHT = 10
HAND_RIGHT = 11
HIP_LEFT = 12
KNEE_LEFT = 13
ANKLE_LEFT = 14
FOOT_LEFT = 15
HIP_RIGHT = 16
KNEE_RIGHT = 17
ANKLE_RIGHT = 18
FOOT_RIGHT = 19

nui_skeleton_names = ( \
    'HIP_CENTER', 'SPINE', 'SHOULDER_CENTER', 'HEAD', \
    'SHOULDER_LEFT', 'ELBOW_LEFT', 'WRIST_LEFT', 'HAND_LEFT', \
    'SHOULDER_RIGHT', 'ELBOW_RIGHT', 'WRIST_RIGHT', 'HAND_RIGHT', \
    'HIP_LEFT', 'KNEE_LEFT', 'ANKLE_LEFT', 'FOOT_LEFT', \
    'HIP_RIGHT', 'KNEE_RIGHT', 'ANKLE_RIGHT', 'FOOT_RIGHT' )

nui_skeleton_conn = ( \
    NONE, \
    HIP_CENTER, \
    SPINE, \
    SHOULDER_CENTER, \
    # Left arm 
    SHOULDER_CENTER, \
    SHOULDER_LEFT,  \
    ELBOW_LEFT,  \
    WRIST_LEFT,  \
    # Right arm 
    SHOULDER_CENTER,  \
    SHOULDER_RIGHT,  \
    ELBOW_RIGHT,  \
    WRIST_RIGHT,  \
    # Left leg 
    HIP_CENTER,  \
    HIP_LEFT,  \
    KNEE_LEFT,  \
    ANKLE_LEFT,  \
    # Right leg 
    HIP_CENTER,  \
    HIP_RIGHT,  \
    KNEE_RIGHT,  \
    ANKLE_RIGHT,  \
)


## Saving the Skeleton definition
G = []
G.append(nui_skeleton_names)
G.append(nui_skeleton_conn)


# Classes used to build the model structure
class model:
    def __init__(self, G=None):
        self.connectivity = G
        self.class_priors = []
        self.jointparts   = []

class NBJoint:
    def __init__(self):
        self.means  = []
        self.sigmas = []
        
class LGMJoint:
    def __init__(slf):
        slf.HC_means  = []
        slf.HC_sigmas = []
        slf.betas     = []
        slf.sigmas    = []


def load_dataset(file=None):
    """
      Returns the data, the labels and the person id for each action
    """
    import scipy.io
    
    if file is None:
        ex = scipy.io.loadmat('data/data.mat')
    else:
        ex = scipy.io.loadmat(file)
        
    return ex['data'],ex['labels'],ex['individuals']

def normpdf(x, mu, sigma):
    """
        Computes Normal PDF
    
    """
    p = (1/(sigma*np.sqrt(2*np.pi)))*math.exp(-( ( (x-mu)/sigma )**2 )/2)
    
    return p


def my_normalizator(probs):
    total = 0
    for i in range(0, len(probs)):
        total = total + probs[i]
        
    return (np.divide(probs,total))

def my_cov(x,y,w):
    """
      Useful function for fit_linear_gaussian
    """
    return np.sum(w*x*y)/np.sum(w)-np.sum(w*x)*np.sum(w*y)/np.sum(w)/np.sum(w)

def fit_gaussian(X, W=None):
    """
      Compute the mean and variance of X, 
      You can ignore W for the moment
    """
    mean = np.mean(X)
    sigma = np.std(X)
    #var = std**2
    
    return (mean, sigma)

def fit_linear_gaussian(Y,X,W = None):
    """
    Input:
      Y: vector of size D with the observations for the variable
      X: matrix DxV with the observations for the parent variables
                 of X. V is the number of parent variables
      W: vector of size D with the weights of the instances (ignore for the moment)
      
    Outout:
       The betas and sigma
    """
    
    #Betas
    D = len(X)
    V = len(X[0])
    b = []
    b.append(np.sum(Y)/D)

    for i in range(0, V):
        b.append(np.sum((Y@X[:,i]))/D)
    
    A = np.zeros((V+1, V+1))
    A[0,0] = 1

    for i in range(0, V):
        A[0,i+1] = np.sum(X[:,i])/D

    for i in range(0, V):
        A[i+1,0] = np.sum(X[:,i])/D

    for i in range(0, V):
        for j in range(0, V):
            A[i+1,j+1] = np.sum((X[:,i]*X[:,j]))/D
    
    betas = np.linalg.solve(A,b)
    
    
    #Sigma
    if (W == None):
        W = np.ones(D)

    s1 = my_cov(Y,Y,W)
    s2 = 0
    N = X.shape[1]

    for i in range(1, N+1):
        for j in range(1, N+1):
            s2 = s2 + betas[i]*betas[j]*my_cov(X[:,i-1],X[:,j-1],W)
    
    sigma = np.sqrt(abs(s1 - s2))
    
    return (betas,sigma)


def learn_model(dataset, labels, G=None):
    """
    Input:
     dataset: The data as it is loaded from load_data
     labels:  The labels as loaded from load_data
     Graph:   (optional) If None, this def should compute the naive 
           bayes model. If it contains a skel description (pe 
           nui_skeleton_conn, as obtained from skel_model) then it should
           compute the model using the Linear Gausian Model

    Output: the model
     a (tentative) structure for the output model is:
       model.connectivity: the input Graph variable should be stored here 
                           for later use.
       model.class_priors: containing a vector with the prior estimations
                           for each class
       model.jointparts[i] contains the estimated parameters for the i-th joint

          For joints that only depend on the class model.jointparts(i) has:
            model.jointparts(i).means: a matrix of 3 x #classes with the
                   estimated means for each of the x,y,z variables of the 
                   i-th joint and for each class.
            model.jointparts(i).sigma: a matrix of 3 x #classes with the
                   estimated stadar deviations for each of the x,y,z 
                   variables of the i-th joint and for each class.

          For joints that follow a gausian linear model model.jointparts(i) has:
            model.jointparts(i).betas: a matrix of 12 x #classes with the
                   estimated betas for each x,y,z variables (12 in total) 
                   of the i-th joint and for each class label.
            model.jointparts(i).sigma: as above

    """
    numClasses = 4
    numJoints = 20
    numCoords = 3
    
    M = model(G)
    
    #Priors
    M.class_priors = [1/numClasses,1/numClasses,1/numClasses,1/numClasses]
    
    #Estimated parameters of Joints
    if(G == None):
          ###############
         # Naïve Bayes #
        ###############
        
        #Single variable
        joint = NBJoint()
        j_m = np.zeros((numCoords,numClasses))
        j_s = np.zeros((numCoords,numClasses))
        
        # Joints loop
        for i in range(0,numJoints):
            # Coordinates loop
            for j in range(0,numCoords):
                # Classes loop
                for k in range(0,numClasses):
                    
                    if(k == 3):
                        clase = 8
                    else:        
                        clase = k+1
                    
                    m,s = fit_gaussian(getInstancesValues(dataset,labels,i,j,clase))
                    j_m[j,k] = m
                    j_s[j,k] = s
            
            joint.means.append(deepcopy(j_m))
            joint.sigmas.append(deepcopy(j_s))
            M.jointparts.append(deepcopy(joint))
            
            j_m = np.zeros((numCoords,numClasses))
            j_s = np.zeros((numCoords,numClasses))
            joint.means  = []
            joint.sigmas = []
        
    else:
          #########################
         # Linear Gaussian Model #
        #########################

        # Single variable
        joint = LGMJoint()
        j_m = np.zeros((numCoords,numClasses))
        j_b = np.zeros((numCoords,numClasses,4))
        j_s = np.zeros((numCoords,numClasses))
        
        # Joints loop
        for i in range(0,numJoints):
            # Coordinates loop
            for j in range(0,numCoords):
                # Classes loop
                for k in range(0,numClasses):
                    
                    if(k == 3):
                        clase = 8
                    else:        
                        clase = k+1
                    
                    Y,X = getInstancesAndParentValues(dataset,labels,i,j,clase,G)
                    
                    if(i == 0):
                        #Special case of HIP_CENTER wich has no parents, we use Mean and Variance
                        m,s = fit_gaussian(Y)
                        
                        j_m[j,k] = m
                        j_s[j,k] = s
                    else:
                        b,s = fit_linear_gaussian(Y,X)
                        
                        j_b[j,k] = np.array(b)
                        j_s[j,k] = s
            
            if(i == 0):
                joint.HC_means.append(deepcopy(j_m))
                joint.HC_sigmas.append(deepcopy(j_s))
            else:
                joint.betas.append(deepcopy(j_b))
                joint.sigmas.append(deepcopy(j_s))
                
            M.jointparts.append(deepcopy(joint))
            
            j_b = np.zeros((numCoords,numClasses,4))
            j_s = np.zeros((numCoords,numClasses))
            joint.betas  = []
            joint.sigmas = []

    return M



def getInstancesValues(data, labels, joint,coordinate,clase):
    """
        Input:
            data: dataset
            labels: labesls dataset
            joint: joint number
            coordinat: x, y or z (0,1 or 2)
            clase: class value (1, 2, 3 or 8)
        Output:
            retorno: Vector with values of a specific
                coordinate (x,y or z) given joint and class.
    """
    retorno = []
    
    for l in range(0,len(labels)):
        if(labels[l]==clase):
            retorno.append(data[joint,coordinate,l])

    return (retorno)


def getInstancesAndParentValues(data, labels, joint, coordinate, clase, G):
    """
        Input:
            data: dataset
            labels: labesls dataset
            joint: joint number
            coordinat: x, y or z (0,1 or 2)
            clase: class value (1, 2, 3 or 8)
            G: skeleton structure
        Output:
            retornoY: Vector with values of a specific
            coordinate (x,y or z) given joint and class.
            retornoX: Vector with values form parent joint.
    """
    retornoY = []
    retornoX = []
    
    N = len(labels)
    PJoint = G[1][joint]
    
    for l in range(0,N):
        if(labels[l]==clase):
            retornoY.append(data[joint,coordinate,l])
            if(PJoint == -1):
                #print("HIP_CENTER")
                retornoX.append(data[joint,coordinate,l])
            else:
                retornoX.append(data[PJoint,:,l])           
    
    return (np.array(retornoY),np.array(retornoX))




def classify_instances(instances, model):
    """    
        Input
           instance: a 20x3x#instances matrix defining body positions of
                     instances
           model: as the output of learn_model

        Output
           probs: a matrix of #instances x #classes with the probability of each
                  instance of belonging to each of the classes

        Important: to avoid underflow numerical issues this computations should
                   be performed in log space
    """

    N = instances.shape[2]
    Nclasses = len(model.class_priors)
    probs = []
    
    # We iterate throught instances
    for i in range(0,N):
        # We calculate the probability of instance ith beloging to each class
        probs.append(compute_logprobs(instances[:,:,i],model))
    
    return np.array(probs)


def GetMu(instance,joint, coord, clase, model):
    """
       Input
           instance: single instance
           joint: joint number
           coord: coordinate index (0, 1 or 2)
           clase: class index
           model: as given by learn_model

       Output
           prbs: calculated mean value
    """
    
    if(joint == 0):
        mu = model.jointparts[0].HC_means[0][coord,clase]
    else:

        parentJoint = model.connectivity[1][joint]

        Xp = instance[parentJoint,0]
        Yp = instance[parentJoint,1]
        Zp = instance[parentJoint,2]

        B0 = model.jointparts[joint].betas[0][coord,clase,0]
        B1 = model.jointparts[joint].betas[0][coord,clase,1]
        B2 = model.jointparts[joint].betas[0][coord,clase,2]
        B3 = model.jointparts[joint].betas[0][coord,clase,3]

        mu = B0 + B1*Xp + B2*Yp + B3*Zp

    return mu


def generate_random_lgm_samples(n, betas, sigma):
    """Function to generate random samples for a 
       Linear Gaussian Model
       Input:
           n: Number of samples
           betas: vector with the values the the betas from 0 to k
           sigma: standard deviation
    """
    X = np.random.randn(n,betas.shape[0]-1)
    Y = np.random.randn(n)*sigma + np.sum(X*betas[1:],axis=1)+betas[0]
    return X,Y



### Code for log space computations

def log_normpdf(x, mean, sigma):
    """
      Computes the natural logarithm of the normal probability density function
      
    """
    prob = -np.log(sigma)-(1/2)*(np.log(2*np.pi))-(1/2)*(((x-mean)/sigma)**2)
    
    return prob


def compute_logprobs(example, model):
    """
       Input
           example: a 20x3 matrix defining body positions of one instance
           model: as given by learn_model

       Output
           prbs: a vector of len #classes containing the loglikelihood of the 
              instance
    """
    
    Nclasses = len(model.class_priors)
    Ncoords = 3
    Njoints = 20
    prbs = []
    prb = []

    if(model.connectivity == None):
        
        # We go through the classes
        for c in range(0,Nclasses):
            # Class prior prob
            prb = np.log(model.class_priors[c])
            
            # We multiply by the likelihood of each joint given parent and class.
            # In Log space we add log probabilities
            for j in range(0,Njoints):
                for i in range(0,Ncoords):

                    x = example[j,i]

                    mu  = model.jointparts[j].means[0][i,c]
                    sig = model.jointparts[j].sigmas[0][i,c]
                    
                    prb = prb + log_normpdf(x, mu, sig)
                    
            prbs.append(prb)  

    else:
    
        # We go through the classes
        for c in range(0,Nclasses):
            # Class prior prob
            prb = np.log(model.class_priors[c])

            # We multiply by the likelihood of each joint given parent and class
            # In Log space we add log probabilities
            for j in range(0,Njoints):
                for i in range(0,Ncoords):

                    x = example[j,i]

                    if(j == 0):
                        mu  = model.jointparts[0].HC_means[0][i,c]
                        sig = model.jointparts[0].HC_sigmas[0][i,c]
                    else:
                        mu = GetMu(example,j, i, c, model)
                        sig = model.jointparts[j].sigmas[0][i,c]

                    prb = prb + log_normpdf(x, mu, sig)
        
            prbs.append(prb)

    return np.array(np.exp(normalize_logprobs(prbs)))


def normalize_logprobs(log_probs):
    """
       Returns the log prob normalizes so that when exponenciated
       it adds up to 1 (Useful to normalizes logprobs)
    """
    mm = np.max(log_probs)
    return log_probs - mm - np.log(np.sum(np.exp(log_probs - mm)))


### Code for estiamtor performance evaluation

def Accuracy(predictions, groundTruth):
    """
       Input
           predictions: vector with predicted classes for each instance
           groundTruth: vector with real classes for each instance

       Output
           accuracy: accuracy of prediction
    """
    N = len(predictions)
    y_pred = []
    y_true = groundTruth
    
    for i in range(0,N):
        
        c = np.where(predictions[i] == np.amax(predictions[i]))
        
        if (c[0][0] == 3):
            c = 8
        else:
            c = c[0][0] + 1
        
        y_pred.append(c)
 
    return (accuracy_score(y_true, y_pred,True))


def RandomVect(N, min,max):
    """
       Input
           N: size of vector to generate
           min: minimum value of vector
           max: maximum value of vector

       Output
           retorno: vector with N random (unique) numbers inside range min-max
    """
    
    retorno = []
    finished = False
    ran = 0
    i = 0
    
    while(not finished):
        ran = random.randrange(min, max+1)
        
        if not(ran in retorno):
            retorno.append(ran)
            i = i + 1
            
        if(i == N): finished = True
        
    return retorno


"""
Stratified k-Fold CrossValidation
"""
def GetPerformance(X, y, n_folds = 4, G = None):
    """
       Input
           X: data
           y: labels
           n_folds: number of folds for K-Fold Cross-Validation
           G: skeleton info

       Output
           model: model object eith model with better accuracy
           measure: better accuracy, corresponds to model returned
           measures: vector with accuracies of all Folds performed,
               it has lenght #n_folds.
    """
    
    skf = StratifiedKFold(n_folds)
    models, measures = [], []
    
    X_ = np.zeros(len(y))
    
    for train_index, test_index in skf.split(X_,y):
        
        ## Extract Kfold
        X_train, X_test = X[:,:,train_index], X[:,:,test_index]
        y_train, y_test = y[train_index], y[test_index]
                
        ## Fit models
        M = learn_model(X_train,y_train, G)
        models.append(learn_model(X_train,y_train, G))
        ## Compute measure
        proba_m = classify_instances(X_test, M)
        #Accuracy(probs, labels_test)
        measures.append(Accuracy(proba_m, y_test))

    i = np.argmax(measures)
    model, measure = models[i], measures[i]
    return model, measure, measures

