import matplotlib.pyplot as plt
from matplotlib.backends import backend_pdf

def plot_motif(motif, time_series, m, save_name, color_string):
    motif_length=motif[0][0][0]-motif[0][-1][0]
    if motif[2].seasonality>0:
        motif_length=int(max(0,motif[2].seasonality-m))
    locations=motif[1]
    fig, axs=plt.subplots(3)
    fig.suptitle(save_name, fontsize=10)
    axs[0].plot(time_series,color="b")
    axs[0].set_title("seasonality is "+str(motif[2].seasonality))
    cap = motif[0][0][-1] - motif[0][0][0]
    if cap <= motif_length+m:
        motif_length=int(min(motif_length,cap-1))
        m = cap-motif_length
    counter=0
    for loc in locations:
        start_loc = int(max(loc - motif_length, 0))
        end_loc = int(min(loc + m, len(time_series)))
        xvalues = [i for i in range(start_loc, end_loc)]
        time_series_values = time_series[start_loc:end_loc]
        
        if len(time_series_values)==1:
            axs[0].plot(xvalues[0],time_series_values[0],marker='o')
        else:
            axs[0].plot(xvalues,time_series_values,color=color_string[counter%len(color_string)])
            counter=counter+1
    counter=0
    for loc in locations:
        start_loc = int(max(loc - motif_length, 0))
        end_loc = int(min(loc + m, len(time_series)))
        time_series_values = time_series[start_loc:end_loc]
        xvalues = [i for i in range(int(start_loc - loc + motif_length), int(end_loc - loc + motif_length))]
        if len(time_series_values) == 1:
            axs[1].plot(xvalues[0], time_series_values[0], marker='o')
        else:
            axs[1].plot(xvalues,time_series_values,color=color_string[counter%len(color_string)])
            counter=counter+1
    axs[2].plot(time_series,color="b")
    proof_locations=[]
    for point in motif[0]:
        proof_locations.append(point[0])
    for proof_point in proof_locations:
        axs[2].axvline(x=proof_point,alpha=0.3,color='r')
    proof_locations=[]
    for point in motif[0]:
        proof_locations.append(point[1])
    for proof_point in proof_locations:
        axs[2].axvline(x=proof_point,alpha=0.3,color='g')
    return fig

def plot_time_series(time_series, save_name):
    fig, axs=plt.subplots(3)
    fig.suptitle(save_name, fontsize=10)
    axs[0].plot(time_series,color="b")
    return fig

def plot_all_motifs(time_series, motifs, m, output_folder, output_file_name):
    fig=[]
    i=0
    color_string=['Aqua','Bisque','Brown','Chartreuse','Coral','Crimson', 'Cyan','DarkGray','DarkMagenta','DarkGoldenrod',
                 'DarkRed','DarkSlateGray','ForestGreen','Fuchsia','Gainsboro','GreenYellow','Gold','Orange','Lavender',
    'LightSteelBlue','MediumPurple','PaleVioletRed','RosyBrown','Teal','Yellow']
    if len(motifs)>8:
        motifs=motifs[:8]
    if len(motifs) == 0:
        figure = plot_time_series(time_series, output_file_name + ": no motif")
        fig.append(figure)
        plt.close(figure)
    for motif in motifs:
        figure=plot_motif(motif,time_series,m,output_file_name + ": motif "+str(i),color_string)
        fig.append(figure)
        plt.close(figure)
        i=i+1
    output_file_index=1
    counter=0
    pdf = backend_pdf.PdfPages(output_folder + output_file_name + ".pdf")
    for figure in fig:
        counter+=1
        if counter%15==0:
            pdf.close()
            output_file_index+=1
            pdf = backend_pdf.PdfPages(output_folder + output_file_name + str(output_file_index) + ".pdf")
        pdf.savefig(figure)
    pdf.close()