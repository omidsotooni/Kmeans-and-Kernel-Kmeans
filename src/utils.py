def squaredDistance(vec1, vec2):
    sum = 0 
    dim = len(vec1) 
    vec2=(np.asarray(vec2)).flatten()
    for i in range(dim):
        sum += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]) 
    
    return sum

#---------- utils done