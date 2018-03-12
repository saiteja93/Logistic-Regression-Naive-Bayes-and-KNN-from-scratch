#Logistic Regression and Naive Baayesian Implementation
#Author : Saiteja Sirikonda


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
import numpy as np
import matplotlib.pyplot as p
np.set_printoptions(precision=6)
import scipy.io as a

#This part of the code computes the dataterm matrix for the given farm advertisement data
with open("farm-ads.txt","r") as file:
	words = file.readlines()


vector = CountVectorizer()
result = vector.fit_transform(words)	

Final = result.toarray()
y=a.savemat('RESULT_Term.mat',mdict={'arr': Final})
			
#defining label vector

label = np.loadtxt("farm-ads-label.txt",skiprows=0, unpack=True)
label_y = np.asarray(label,int)
Y = label_y[1]
#Y = Y[:3000]
#print("Y shape:\t",Y.shape)

class Naive_Bayes_Classification(object):

	def __init__(self, p = 1.0):
		self.p = p

	def probability(self, X, Y):
        #This part of the code, basically helps in computing the class prior by separting the data points,
        #different class labels
		separator = [[attributes for attributes,temp in zip(X,Y) if temp==classl] for classl in np.unique(Y)]
		total_samples = X.shape[0]
		self.prior = [np.log(len(i)/total_samples) for i in separator]				#computing the prior log probability for each class
		total_words = np.array([np.array(k).sum(axis=0) for k in separator]) + self.p#count of each word for each class
		
                        
		self.ll = np.log((total_words)/(total_words.sum(axis=1)+43602)[np.newaxis].T)	
		return self

	def probability_prediction(self,X): #Calculate using the Naive Bayes Assumption
        
		prob = ([(self.ll * x).sum(axis=1) + self.prior for x in X])
		return prob

	def maximum(self, X): #Predicting the majority class label for the given sample
		maxi = np.argmax(self.probability_prediction(X), axis=1)				#calls predict_log probabilities and picks maximum value
		
		return maxi

	def total(self, X, Y):
		return (sum(self.maximum(X) == Y)/len(Y))
    
   
def LR(data, labels, steps, rate, c = False):
    if c:
        cept = np.ones((data.shape[0], 1))
        data = np.hstack((cept, data))
        
    w_vector = np.zeros(data.shape[1])
    
    for step in range(steps):
        total = np.dot(data, w_vector)
        estimate = sigmoid(total)

        # Update weights with gradient
        error = labels - estimate
        difference = np.dot(data.T, error)
        w_vector += rate * difference
        
        # Print log-likelihood every so often
        if step % 10000 == 0:
            print (ll(data, labels, w_vector))
    
    return w_vector

def sigmoid(total):
    return 1 / (1 + np.exp(-total))

def ll(data, labels, w_vector):
    total = np.dot(data, w_vector)
    loglikely = np.sum( labels*total - np.log(1 + np.exp(total)) )
    return loglikely


Naive_accuracy=[]
LR_accuracy=[]
ratios = [0.1,0.3,0.5,0.7,0.8,0.9]
"""for k in ratios:    
    X_train, X_test, Y_train, Y_test = train_test_split(Final, Y, test_size=1-k, random_state = True)
    
    naive = Naive_Bayes_classification().probability(X_train, Y_train)    
    Naive_accuracy.append(nb.total(X_test, Y_test))
    weights = LR(X_train, Y_train,num_steps =200 , learning_rate = 6e-6, add_intercept=True)
    data_intercept = np.hstack((np.ones((X_test.shape[0], 1)),X_test))
    final = np.dot(data_intercept, weights)
    prediction = np.round(sigmoid(final))
    LR_accuracy.append((prediction == Y_test).sum().astype(float) / len(prediction))
p.plot(ratios,Naive_accuracy,'r')
p.plot(ratios,LR_accuracy,'b')
p.xlabel('Fraction of records in the Training set')
p.ylabel('Accuracy')
"""

for k in ratios:
    Naive_accuracy_5_times =[]
    LR_accuracy_5_times =[]
    for i in range(5):
        X_train, X_test, Y_train, Y_test = train_test_split(Final, Y, test_size=1-k)
        naive = Naive_Bayes_Classification().probability(X_train, Y_train)
        Naive_accuracy_5_times.append(naive.total(X_test, Y_test))
        weights = LR(X_train, Y_train, steps =200 , rate = 6e-6, c=True)
        data_intercept = np.hstack((np.ones((X_test.shape[0], 1)),X_test))
        final = np.dot(data_intercept, weights)
        prediction = np.round(sigmoid(final))
        LR_accuracy_5_times.append((prediction == Y_test).sum().astype(float) / len(prediction))
    Naive_accuracy.append(np.mean(Naive_accuracy_5_times, axis = 0))
    LR_accuracy.append(np.mean(LR_accuracy_5_times, axis = 0))
p.plot(ratios,Naive_accuracy,'r')
p.plot(ratios,LR_accuracy,'b')
p.xlabel('Fraction of records in the Training set')
p.ylabel('Accuracy')
        
        
        
        
        








