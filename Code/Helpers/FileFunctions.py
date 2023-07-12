import glob
import pandas as pd
from PyPDF2 import PdfMerger 

def merge_pdfs(folder):
    pdfs = glob.glob(folder + "*.pdf")
    
    merger = PdfMerger()
    
    for pdf in pdfs:
        merger.append(pdf, pages=(0, 1))
    
    merger.write(folder + "result.pdf")
    merger.close()

def load_csv_files(folder):
    output = []

    files = glob.glob(folder + "*.csv")
    for f in files:
        output.append(pd.read_csv(f))

    return output, files


def get_files(folder, extension="csv"):
    return glob.glob(folder + "*." + extension)

def load_file(file, header=0, delimiter=","):
    return pd.read_csv(file, delimiter=delimiter, header=header)