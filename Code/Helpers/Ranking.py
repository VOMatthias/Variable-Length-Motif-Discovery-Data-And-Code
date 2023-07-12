import numpy as np
from collections import namedtuple

motifPropertyStruct=namedtuple("motifPropertyStruct","Prominence, seasonality")

def try_expand(time_series, start, stop, new_length):
    if stop-start>=new_length:
        return start,stop
    extension_needed= new_length-(stop-start+1)
    halfextension_needed=extension_needed/2+1
    # we will try to look at the motif of double length
    extension_room = min(len(time_series) - stop-1, halfextension_needed)  
    stop = stop + extension_room
    extension_needed=new_length-(stop-start+1)
    extension_room = min(start, extension_needed)
    start -= extension_room
    if extension_room!=extension_needed:
        extension_room=extension_needed-extension_room
        stop=stop+min(extension_room,len(time_series)-stop-1)
    return int(start),int(stop)

def get_motif_properties(time_series,motif,locations,m):
    prominence=calculate_prominence(time_series,motif,locations,m)
    start=motif[-1][0]
    stop=min(motif[0][0]+m,len(time_series))
    #144=half a day: we want to see the seasonality for at least half a day? or should it be half a week?
    start,stop=try_expand(time_series,start,stop,144) 
    seasonality=find_seasonality(time_series, start,stop)
    # try taking double the motif: seasonal stuff doesn't score well with prominence, so we need to make sure 
    # we don't miss any periods
    length = 2*(motif[0][0]-motif[-1][0]+m)
    #144=half a day: we want to see the seasonality for at least half a week
    #(stop-start) because we don't want out period to be larger than distance to diagonal
    start,stop=try_expand(time_series,start,stop,max(length,144)) 
    seasonality=max(find_seasonality(time_series,start,stop),seasonality) 
    #added a term "+m" because it happened that a motifproof was found on diagonal 285 and
    #it really actually was a daily seasonality (ie 288 period). The "m" can be replace by +10 or something as well.    
    if seasonality>np.abs(motif[0][0]-motif[0][1])+m: 
        seasonality=0
    if seasonality!=0 and 3*seasonality<stop-start and seasonality<144:
        start,stop=try_expand(time_series,start,stop,length/2*3)
        seasonality=find_seasonality(time_series,start,stop)
    return motifPropertyStruct(prominence,seasonality)

def detrend(time_series):
    slopes=[(time_series[-i]-time_series[i])/(len(time_series)-2*i)for i in range(0,int(len(time_series)/3))]
    slope=np.median(slopes)
    return [time_series[i]-i*slope for i in range(0,len(time_series))]

def get_outliers(time_series):
    quartiles=np.quantile(time_series,[0.05,0.5,0.95])
    IQR=quartiles[2]-quartiles[0]
    if IQR==0:
        return [i for i in range(0,len(time_series)) if time_series[i]>quartiles[1]]
    return [i for i in range(0,len(time_series)) if time_series[i]>quartiles[1]+1.5*IQR or time_series[i]<quartiles[1]-1.5*IQR]

def find_seasonality(time_series,start,stop,depth=0):
    corrs = []
    time_series=time_series[start:stop]
    max_lag = int(max(0, min(len(time_series) * 0.5, len(time_series) - 10)))
    for i in range(max_lag):
        corrs.append(np.corrcoef(time_series[0:len(time_series) - i], time_series[i:len(time_series)])[0][1])
    counter=0
    for counter, corr in enumerate(corrs):
        if corr <0.5:
            break
    if(counter==len(corrs)-1): #never goes below 0.5: maybe like an increasing or decreasing line -> detrend
        if depth==0: #apply this function to some cleaner time_series
            time_series=detrend(time_series)
            outlier_positions=get_outliers(time_series)
            for i in range(1,len(outlier_positions)):
                if outlier_positions[i]-outlier_positions[0]>=24:
                    return outlier_positions[i]-outlier_positions[0]
            return find_seasonality(time_series,0,len(time_series),1)
        else:
            return 0
    high_correlation_range=[]
    high=False
    expected_seasonalities=np.array([168,288,2016,4032]) #one hour, a day, a week, 2 weeks
    period_results=[]
    maximum=0
    for c in range(counter,len(corrs)):
        #close_to_expected_season= np.min(np.abs(expected_seasonalities-c))<min(6,c/12)
        if(corrs[c]>maximum):
            maximum=corrs[c]
        if corrs[c]>0.95-high*0.01:#-0.2*close_to_expected_season:
            high_correlation_range.append(c)
            high=True
        elif high:
            high=False
            period_results.append([int(np.mean(high_correlation_range)),maximum])
            maximum=0
    if high:
        period_results.append([int(np.mean(high_correlation_range)), maximum])
    if len(period_results)==0:
        return 0
    maximum=0
    for result in period_results:
        if(maximum<result[1]):
            maximum=result[1]
    for result in period_results:
        if result[1]>=maximum-0.025:
            return result[0]

def calculate_prominence(time_series,motif,locations,m):
    motif_length = motif[0][0] - motif[-1][0]
    cap = motif[0][-1] - motif[0][0]
    if cap <= motif_length + m:
        motif_length = min(motif_length, cap - 1)
        m = cap - motif_length
    stdevs=[]
    for loc in locations:
        start_loc = max(loc - motif_length, 0)
        end_loc = min(loc + m, len(time_series))
        time_series_values_long = time_series[start_loc:end_loc]
        time_series_values_short =time_series[start_loc:int(min(len(time_series),max(loc+2,start_loc+5)))]
        if len(time_series_values_long)>1:
            if len(time_series_values_short)>=5:
                stdevs.append(max(np.std(time_series_values_short), np.std(time_series_values_long)))
            else:
                stdevs.append(np.std(time_series_values_long))
    return np.mean(stdevs) if len(stdevs)>0 else 0

def calculate_total_prominence(time_series,length):
    non_zero_prominences=[]
    prominences=[]
    for i in range(len(time_series)-length):
        stdev=np.std(time_series[i:i+length])
        prominences.append(stdev)
        if stdev!=0:
            non_zero_prominences.append(stdev)
    # if basically everything is constant, let everything be a motif and just return the most extreme one.
    if len(non_zero_prominences)<0.1*(len(time_series)-length): 
        return [0,0,0]
    else:
        quantiles=np.quantile(prominences,[0.25,0.5,0.75])
        return quantiles

def order_motifs_on_prominence(time_series,motifs,m):
    prominences=[]
    ordered_motifs = []
    if len(motifs)==0:
        return ordered_motifs, prominences
    for motif in motifs:
        start=motif[0][-1][0]
        stop=motif[0][0][0]
        length=int(len(time_series)/3)
        start,stop=try_expand(time_series,start,stop,int(max(length,110)))
        reference_stdev=calculate_total_prominence(time_series[start:stop],100)
        if reference_stdev[0]==reference_stdev[2]: #a lot of constant time_series, everything should be a motif
            if motif[2].Prominence >reference_stdev[1]:
                prominences.append(motif[2].Prominence+1.5) #make sure everything is a motif, meaning it should have value >threshold=1.5
            else:
                prominences.append(0)
        else:
            prominences.append((motif[2].Prominence-reference_stdev[1])/(reference_stdev[2]-reference_stdev[0]))
    maxP=np.max(np.abs(prominences))
    prominences_new=[]
    for i, motif in enumerate(motifs):
        new_entry=[prominences[i],prominences[i]]
        if motif[2].seasonality>0:
            prominences[i]+=2*max(maxP,1.5)+0.01 #make sure this gets the highest prominence and also make sure it is not ignored (ie>median + 1.5*IQR)
            new_entry=[new_entry[0],prominences[i]]
        prominences_new.append(new_entry)
    ordered_indices=np.argsort(prominences)
    ordered_motifs=[motifs[i] for i in list(reversed(ordered_indices))]
    prominences= ["Ref "+str(reference_stdev) +" new "+str(prominences_new[i][1]) + " old "+str(prominences_new[i][0]) for i in list(reversed(ordered_indices))]
    return ordered_motifs, prominences