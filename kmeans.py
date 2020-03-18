import numpy as np





def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data-  numpy array of points
    :param generator: random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being index to a sample
             which is chosen as centroid.
    '''
    # TODO:
    # implement the Kmeans++ algorithm of how to choose the centers according to the lecture and notebook
    # Choose 1st center randomly and use Euclidean distance to calculate other centers.
    #raise Exception(
             #'Implement get_k_means_plus_plus_center_indices function in Kmeans.py')
    
    
    centers = []
    centers.append(generator.randint(0, n))
    e = []
    for i in range(n_cluster):
    #while len(centers) < n_cluster:
        #d2 = dist_from_centroids(centers[-1], x, d)
        disto=e
        result1 = []
        i=0
        while i<len(x):
        #for i in range(len(x)):
            if len(disto)==0:
                result=[]
            else:    
                result = [disto[i]]
            result.append((np.linalg.norm(x[i] - x[centers[-1]]))**2)
            result1.append(min(result))
            i+=1
        disto=result1     
        
        centers.append(np.argmax(disto/sum(disto)))
        e = disto
    

    # DO NOT CHANGE CODE BELOW THIS LINE

    print("[+] returning center for [{}, {}] points: {}".format(n, len(x), centers))
    return centers



def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)




class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a length (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates a Int)
            Note: Number of iterations is the number of time you update the assignment
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
       # raise Exception(
           #  'Implement fit function in KMeans class')
        
        
        # Initialize centroids, membership, distortion
        mu = x[self.centers,:]#np.random.choice(N, self.n_cluster, replace=True), :]
        rem = np.zeros(N)
        J = 999999
        
        # Loop until convergence/max_iter
        iter = 0
        #for i in range(self.max_iter):
        while iter < self.max_iter:
            d2 = np.empty((0,N),float)
            J_new = 0
            for i in range(self.n_cluster):
                dist = mu[i] - x
                l2 = np.sum(dist * dist, axis = 1)
                d2 = np.append(d2, [l2], axis = 0)
            rem = np.argmin(d2, axis = 0)
            for i in range(self.n_cluster):
                J_new += np.sum( d2[i,:]* np.array(rem == i) )                       
                        
            J_new /= N
            
            
            
            if np.absolute(J - J_new) <= self.e:
                #return mu, r, i
                break
            J = J_new
            # Compute means
            mu_new = np.array([np.mean(x[rem == k], axis=0) for k in range(self.n_cluster)])
            index = np.where(np.isnan(mu_new))
            mu_new[index] = mu[index]
            mu = mu_new
            iter += 1
            
         #print(mu,self.centers)    
        #return mu, r, self.max_iter
        centroids=mu
        y = rem
           
        
        # DO NOT CHANGE CODE BELOW THIS LINE
        return centroids, y, self.max_iter

        


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)

            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (N,) numpy array)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        #raise Exception(
            # 'Implement fit function in KMeansClassifier class')
        k_means = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e,generator=self.generator)
        centroids, membership, _ = k_means.fit(x)
        polls = [{} for k in range(self.n_cluster)]
        temp=np.array((y,membership)).T
        #for y_i, r_i in zip(y, membership):
        for p in range(len(temp)):
            if temp[p][0] not in polls[temp[p][1]].keys():
                polls[temp[p][1]][temp[p][0]] = 1
            else:
                polls[temp[p][1]][temp[p][0]] += 1
        centroid_labels = []
        for polls_k in polls:
            if not polls_k:
                centroid_labels.append(0)
            centroid_labels.append(max(polls_k, key=polls_k.get))
        centroid_labels = np.array(centroid_labels)
        
       
        

        
        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        #raise Exception(
           #  'Implement predict function in KMeansClassifier class')
            
        #cj=[]
        lab_ind = np.argmin(np.linalg.norm(x[:, None]-self.centroids, axis=2), axis=1)
        labels=self.centroid_labels[lab_ind]
        
     
        
        
        
        # DO NOT CHANGE CODE BELOW THIS LINE
        return np.array(labels)
        

def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors

        Return new image from the image by replacing each RGB value in image with nearest code vectors (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    # TODO
    # - comment/remove the exception
    # - implement the function

    # DONOT CHANGE CODE ABOVE THIS LINE
    #raise Exception(
           #  'Implement transform_image function')
    #def cal_argmin(a):
        
        
    P, M, D = image.shape
    content = image.reshape(P * M, D)
    #l2 = np.sum(((content - np.expand_dims(code_vectors, axis=1)) ** 2), axis=2)
    reg = np.argmin(np.sum((np.square(content - np.expand_dims(code_vectors, axis=1))), axis=2), axis=0)
    new_im= code_vectors[reg].reshape(P, M, D)
    
    #def argmin(a, axis=0):
       # return min(range(len(a))
    #argmin(a) 

    
    

    # DONOT CHANGE CODE BELOW THIS LINE
    return new_im

#if __name__=="__main__":
   # get_k_means_plus_plus_center_indices(5,4,np.array([0,1,2,3,4]).reshape(5,1))
