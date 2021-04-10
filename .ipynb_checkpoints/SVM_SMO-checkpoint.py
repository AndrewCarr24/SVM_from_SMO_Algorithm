import numpy as np
from numba import jit

class SVM():
    
    def __init__(self, max_iter=1000, kernel='linear', C=1.0, epsilon=0.001, gamma = 5):

        self.max_iter = max_iter
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.kernel = kernel
        
    # Train SVM model using SMO algorithm
    def fit(self, X, y):
        # Init
        alphas = np.ones(len(X)).reshape(-1, 1)
        bias = 0
        
        for i in range(self.max_iter):
    
            alphas_prev = alphas.copy()
            
            for main_idx, item in enumerate(alphas):
                
                other_idx = np.random.choice(np.where(np.arange(X.shape[0]) != main_idx)[0], 1)[0]
            
                if self.eta_func(X, main_idx, other_idx) == 0:
                    continue
                
                candidate_alpha = self.alph_update(X, y, main_idx, other_idx, alphas, bias)
                    
                candidate_alpha = self.apply_bounds(y, alphas, main_idx, other_idx, candidate_alpha)
                  
                candidate_other = self.other_from_main(y, alphas, main_idx, other_idx, candidate_alpha)
            
                bias_other_new = self.update_bias(X, y, other_idx, main_idx, alphas, bias, candidate_alpha, candidate_other)
                bias_main_new = self.update_bias(X, y, main_idx, other_idx, alphas, bias, candidate_alpha, candidate_other)
                
                if candidate_other > 0 and candidate_other < self.C:
                    bias = bias_other_new
                elif candidate_alpha > 0 and candidate_alpha < self.C:
                    bias = bias_main_new
                else:
                    bias = (bias_main_new + bias_other_new)/2
                    
                alphas[main_idx] = candidate_alpha
                alphas[other_idx] = candidate_other
                
            diff = np.sum((alphas - alphas_prev)**2)
            if diff < self.epsilon:
                break
                
        self.alpha = alphas
        # self.bias = bias
        
        w_fin = np.sum(X * (y * alphas.reshape(-1)).reshape(-1, 1), axis = 0)
        self.w = w_fin
        
        if self.kernel == 'linear':
            self.bias = np.mean(y - np.dot(w_fin.reshape(1,-1),X.T))
        else:
            wx = []
            for k in range(len(X)):
                wx.append(np.sum(alphas.reshape(-1) * y * self.K(X[k], X)))
            wx = np.array(wx)
            self.bias = np.mean(y - wx)
        
        self.train_X = X
        self.train_y = y
         
    # Calling method for K_numba
    def K(self, x1, x2):
        gamma = self.gamma
        kernel = self.kernel
        return self.K_numba(x1, x2, kernel, gamma)
      
    # Kernel function (optimized with numba)
    @staticmethod            
    @jit(nopython=True)
    def K_numba(x1, x2, kernel, gamma):
        
        out_lst = []
        
        if(len(x2.shape) == 1):
            x2 = x2.reshape(1, -1)
            
        if kernel == "linear":
              
            for i in x2:
                out_lst.append(np.sum(x1 * i))
                    
        elif kernel == "rbf":
            
            for i in x2:
                out_lst.append(np.exp(-1*gamma*(np.sum((x1 - i)**2))))
            
        return np.array(out_lst)
    
    # Update alpha_j
    def alph_update(self, X, y, main_idx, other_idx, alphas, bias):
        return alphas[main_idx][0] - y[main_idx]*(self.E_func(other_idx, X, y, alphas, bias) - self.E_func(main_idx, X, y, alphas, bias))/self.eta_func(X, main_idx, other_idx)
    
    # Helpers for alph_update
    def pred_func(self, idx, X, y, alphas, bias):
        return np.sum(alphas.reshape(-1) * y * self.K(X[idx], X)) + bias
        
    def E_func(self, idx, X, y, alphas, bias):
        return self.pred_func(idx, X, y, alphas, bias) - y[idx]
    
    def eta_func(self, X, main_idx, other_idx):
        return np.array([np.sum(2*self.K(X[main_idx], X[other_idx].reshape(1, -1)) - self.K(X[main_idx], X[main_idx].reshape(1, -1)) - self.K(X[other_idx], X[other_idx].reshape(1, -1)))])
    
    
    # Adjusting alpha_j to fall between L and H
    def apply_bounds(self, y, alphas, main_idx, other_idx, cand_alpha):
        
        s = y[main_idx] * y[other_idx]
        
        if s == 1:
            L = np.max([0, alphas[main_idx][0] + alphas[other_idx][0] - self.C])
            H = np.min([alphas[main_idx][0] + alphas[other_idx][0], self.C])
        elif s == -1:
            L = np.max([0, alphas[main_idx][0] - alphas[other_idx][0]])
            H = np.min([self.C + alphas[main_idx][0] - alphas[other_idx][0], self.C])
            
        if cand_alpha >= H:
            return H
    
        elif cand_alpha <= L:
            return L
        else:
            return cand_alpha
        
    # Update alpha_i based on new value for alpha_j
    def other_from_main(self, y, alphas, main_idx, other_idx, cand_alpha):
        
        s = y[main_idx] * y[other_idx]
        a_i_old = alphas[other_idx][0]
        a_j_new = cand_alpha
        a_j_old = alphas[main_idx][0]
        
        return a_i_old - (s*(a_j_new - a_j_old))
        
    # Update bias term
    def update_bias(self, X, y, main_idx, other_idx, alphas, bias, candidate_alpha, candidate_other):
        
        E_old = self.E_func(main_idx, X, y, alphas, bias) 
        main_pt = y[main_idx]*(candidate_alpha - alphas[main_idx]) * self.K(X[main_idx], X[main_idx].reshape(1, -1))
        other_pt = y[other_idx]*(candidate_other - alphas[other_idx]) * self.K(X[other_idx], X[main_idx].reshape(1, -1))
        
        return bias - (E_old + main_pt + other_pt)
    
    # Predictions based on new values
    def predict(self, X):
        
        if self.kernel == "linear":
            return np.sign(np.dot(self.w, X.T) + self.bias)
        elif self.kernel == "rbf":
            wx = []
            for k in range(len(X)):
                wx.append(np.sum(self.alpha.reshape(-1) * self.train_y * self.K(X[k], self.train_X)))
            wx = np.array(wx)
            # bias = np.mean(self.train_y - wx)
            return np.sign(wx + self.bias)
                
        