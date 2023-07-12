import MatrixProfile
import numpy as np
from itertools import groupby
from statsmodels.tsa.stattools import acf

def detrend(time_series):
    slopes=[(time_series[-i]-time_series[i])/(len(time_series)-2*i)for i in range(0,int(len(time_series)/3))]
    slope=np.median(slopes)
    return [time_series[i]-i*slope for i in range(0,len(time_series))]

def get_periodicity(input_series, threshold=0.7):
    #print(input_series)
    acf_signal = acf(detrend(input_series),nlags=len(input_series))
    
    below = np.where(acf_signal<threshold)
    ranges = get_submotifs(below[0])
    if(len(ranges)>1):
        periodicity_candidates = np.where(acf_signal==acf_signal[ranges[0][-1]+1:ranges[1][0]].max())
        candidate_index = periodicity_candidates[np.where(periodicity_candidates>ranges[0][-1])[0][0]][0]
    else:
        candidate_index = -1
    return candidate_index

def adjusted_hampel_filter(input_series, n_sigmas=3):
    k = 1.4826 # scale factor for Gaussian distribution
    
    X = [i for i in input_series if i != 0]
    x0 = np.median(X)
    S0 = k * np.median(np.abs(X - x0))
    temp = input_series - x0 < -n_sigmas * S0
            
    return np.where(temp == True)[0]

def shift(arr, fill_value):
    # make diagonals rows and fill the missing values with fill_value    
    size0 = np.size(arr, 0)
    size1 = np.size(arr, 1)
    
    result = np.ones((size0+size1-1, size1))*fill_value
    
    for i in range(size1):
        result[size1-1-i:size0+size1-1-i, i] = arr[:, i]
    
    return result

def get_submotifs(seq):
    # group the consecutive numbers
    return [
        [x for _, x in g]
        for k, g in groupby(
            enumerate(seq), 
            lambda i_x : i_x[0] - i_x[1]
        )
    ]

def get_diagonals_sorted(bin_matrix, m):
    # sort the diagonals according to the number of ones on the diagonal
    n = len(bin_matrix)
    temp = shift(bin_matrix, fill_value=0)
    temp = temp[:n - m//2]
    temp = np.sum(temp, axis=1)
    return np.argsort(temp)[::-1]

def get_motifs(time_series, m, threshold):
    output = []
    
    print("Calculating matrix")
    profile, bin_matrix = MatrixProfile.stomp(time_series, m, threshold)
        
    print("Processing matrix")
    # go over each diagonal (order according to the number of events on the diagonal)
    range_diag = get_diagonals_sorted(bin_matrix, m)
    i = 0
    for x in range_diag:
        if i%100==0:
            print("\t",i, "diagonals processed")
        i += 1
        
        motifs, binary_matrix = process_diagonal(x, time_series, bin_matrix, m, threshold, profile)

        output += motifs
    return output

def process_diagonal(row_nr, time_series, bin_matrix, m, threshold, profile):
    output = []
    motif = []
    
    n = len(bin_matrix)
    max_motif_size = n-row_nr-1
    
    y = row_nr
    x = n-1
    
    while(y>=0):
        if(bin_matrix[y,x]==1):
            #first event found
            if(len(motif) == 0): 
                motif.append([y,x])

                # look to the left
                for i in range(1, max_motif_size):
                    if profile[x - i] < threshold and y - i >= 0:
                        motif.append([y - i, x - i])

                # look to go back right
                for i in range(1, max_motif_size - (motif[0][0] - motif[-1][0]) - 1):
                    if x + i < len(profile) and profile[x + i] < threshold:
                        motif.insert(0, [y+i, x+i])
                
                y -= max_motif_size
                x -= max_motif_size
                
                #check if good motif proof is found
                motif_length=motif[0][0]-motif[-1][0]
                score = MatrixProfile.calculate_scaled_match_distance(time_series, motif[-1][0], motif[-1][1], motif_length + m)
                
                motifs = []              
                #check if subparts do form a good motif
                if(score > threshold):
                    series = profile[motif[-1][0]:motif[-1][0] + motif_length + 1]
                    indices = get_submotifs(adjusted_hampel_filter(series, n_sigmas=2))
                    merged_indices = []
                    if len(indices) != 0:
                        index = indices.pop(0)
                        while len(indices) > 0:
                            index2 = indices[0]
                            motif_length=index2[-1]-index[0]
                            score = MatrixProfile.calculate_scaled_match_distance(time_series, index[0]+motif[-1][0], index[0]+motif[-1][1], motif_length + m)
                            if score < threshold:
                                index += index2
                            else:
                                merged_indices.append(index)
                                index = index2

                            indices.pop(0)
                           
                        merged_indices.append(index)
                        
                        for index in merged_indices:
                            index = np.array(index)
                            temp = np.array([index+motif[-1][0], index+motif[-1][1]]).T[::-1]
                            if 1 in bin_matrix[temp.T[0], temp.T[1]]:
                                motifs.append(temp)
                else:
                    motifs = [motif]
                
                for mot in motifs:
                    #length_mot = mot[0][0]-mot[-1][0]
                    #
                    ##print(mot)
                    ##print(length_mot)
                    #
                    #periodicity = get_periodicity(time_series[mot[-1][0]:mot[-1][0]+length_mot])
                    #if(periodicity>0):
                    #    #periodicity found in motif
                    #    for i_motif in range(len(mot)):
                    #        if(mot[i_motif][0] - mot[-1][0] > periodicity):
                    #            mot.pop()
                    motif_indices, bin_matrix = fetch_motifs(time_series, m, bin_matrix, mot, threshold)
                    motif_indices = list(motif_indices)
                    
                    output.append([mot,motif_indices])
                    
                motif=[]
    
        y-=1
        x-=1

    return output, bin_matrix

def fetch_motifs(time_series, m, bin_matrix, motif_proof, threshold):
    motif_length = motif_proof[0][0] - motif_proof[-1][0]
      
    d_l_y = MatrixProfile.calculate_distance_profile(time_series, motif_proof[-1][0], motif_length + m)  
    d_l_x = MatrixProfile.calculate_distance_profile(time_series, motif_proof[-1][1], motif_length + m)

    proof_y = convert_proof_distances_to_binary(d_l_y, threshold)
    proof_x = convert_proof_distances_to_binary(d_l_x, threshold)

    indices_x = np.where(proof_x == 1)[0]
    indices_y = np.where(proof_y == 1)[0]
        
    indices = extract_motif_ranges(indices_x,indices_y,motif_length, d_l_y, d_l_x) 
    output_indices = set(indices)

    bin_matrix = exclude_ranges(output_indices, motif_length+1, bin_matrix)
    
    return output_indices, bin_matrix

def exclude_ranges(indices, motif_size, bin_matrix):
    # excludes the ranges of the vertical and horizontal matches of a motif in the binary matrix
    for x in indices:
        for y in indices:
            bin_matrix[max(y-motif_size+1,0):y+1,:] = 0
            bin_matrix[:, max(x-motif_size+1,0):x+1] = 0        

    return bin_matrix

def extract_motif_ranges(indices_x, indices_y, motif_length, d_l_y, d_l_x, size_threshold=0.9):
    output = []
    indices = set(indices_x)
    indices = indices.union(indices_y)
    indices = sorted(indices)
    
    proof = np.minimum(d_l_y, d_l_x)
    sortedProof = proof[indices]
    sort = sorted(range(len(sortedProof)), key=lambda k: sortedProof[k])
    
    while len(sort) > 0:
        i = sort.pop(0)
        val = indices[i]
        output.append(indices[i] + motif_length)
        sort = [i for i in sort if abs(indices[i] - val) / (motif_length + 1)> size_threshold]
    
    return output

def convert_proof_distances_to_binary(motif_proof_distances, threshold=0.4):
    # converts the distances of some motif proof along a certain axis to a binary matching matrix
    # (in other words, this returns a list where a 1 means the motif is a match on that index, whereas a 0 is not a match)
    # (we make sure you cannot have multiple matches right after eachother, we scan for clusters of distances below the threshold, 
    # and take the best match as representative of that cluster, ignoring all the other matches in the output list)
    
    output = np.zeros(len(motif_proof_distances))
    pivot = -1
    pivot_collection = []

    for x in range(len(motif_proof_distances)):
        if (motif_proof_distances[x] < threshold):
            pivot_collection.append(motif_proof_distances[x])
        else:
            if (len(pivot_collection) > 0):
                minimum_index = np.argmin(pivot_collection)
                output[pivot + minimum_index + 1] = True
                pivot_collection = []
            pivot = x

    if (len(pivot_collection) > 0):
        minimum_index = np.argmin(pivot_collection)
        output[pivot + minimum_index + 1] = True

    return output