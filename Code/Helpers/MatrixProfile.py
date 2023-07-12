from numba import jit
import numpy as np 
import numba as nb

@jit 
def calculate_distance_profile(T, i, m):
    mu, sigma = compute_mean_std(T,m)
    QT = sliding_dot_product(T[i:i+m],T)
    return calculate_distance_profile_scale(QT, mu, sigma, i, m)

@jit
def calculate_scaled_match_distance(T, start_x_1, start_x_2, m, c = 1.0):
    window_1 = T[start_x_1:start_x_1+m]
    window_2 = T[start_x_2:start_x_2+m]
    
    window_1 = window_1-window_1.mean()
    window_2 = window_2-window_2.mean()
    
    std_1 = window_1.std()
    std_2 = window_2.std()
    
    if(std_1>0):
        window_1 = window_1/std_1
    if(std_2>0):
        window_2 = window_2/std_2
    
    min_std = min(std_1,std_2)
    max_std = max(std_1,std_2)
    
    Dscale = max_std
    if(min_std!=0):
        Dscale = max_std/min_std - 1.0
    
    Dscale = min(Dscale,c)/c
    Dznorm = np.sqrt(((window_1-window_2)**2).sum())/(2*np.sqrt(m))
    
    distance = np.sqrt(Dscale**2+Dznorm**2)/np.sqrt(2)
    
    return distance

@jit
def sliding_dot_product(Q,T):
    n = len(T)
    m = len(Q)
    QT = np.zeros(n-m+1)
    for i in range(0,n-m+1):
        QT[i] = np.dot(Q, T[i:i+m])
    return QT

@jit
def calculate_distance_profile_scale(QT, mu, sigmas, i, m, c = 2.0):
    sigma_max = np.copy(sigmas)
    sigma_i = sigmas[i]
    sigma_max[sigma_max < sigma_i] = sigma_i
    sigma_min = np.copy(sigmas)
    sigma_min[sigma_min > sigma_i] = sigma_i
    Dscale = sigma_max/sigma_min - 1
    Dscale[sigma_min == 0] = sigma_max[sigma_min == 0]
    Dscale[Dscale > c] = c
    Dscale /= c
    sigma = np.copy(sigmas)
    mu_i = mu[i]
    sigma[sigma == 0] = 1 #to avoid nans in the output
    sigma_i = sigma[i]
    Dznorm = np.sqrt(2*m*(1-(QT - m*mu_i*mu)/(m*sigma_i*sigma)))/(2*np.sqrt(m))
    return np.sqrt(Dznorm**2+Dscale**2)/np.sqrt(2)

@jit
def compute_mean_std(T,m):
    n = len(T)
    l = n-m+1
    mu = np.ones(l)
    sigma = np.ones(l)
    for i in range(0, l):
        mu[i] = np.mean(T[i:i+m])
        sigma[i] = np.std(T[i:i+m])
    return mu, sigma

@jit
def stomp(T,m,threshold):
    n = len(T)
    l = n-m+1
    M = np.ones((l,l))
    mu, sigma = compute_mean_std(T,m)
    QT = sliding_dot_product(T[0:m],T)
    QT_first = np.copy(QT)
    D = calculate_distance_profile_scale(QT, mu, sigma, 0, m)
    D[0:m//4] = np.ones(m//4)
    M[0] = D < threshold
    P = np.copy(D)
    for i in range(1,l):
        for j in range(l-1,0,-1):
            QT[j] = QT[j-1]-T[j-1]*T[i-1]+T[j+m-1]*T[i+m-1]
        QT[0] = QT_first[i]
        D = calculate_distance_profile_scale(QT, mu, sigma, i, m)
        start = max(i-m//4,0)
        end = min(i+m//4, l)
        D[start:end] = np.ones(end - start)
        M[i] = D < threshold
        P = np.minimum(P,D)
    return P, M