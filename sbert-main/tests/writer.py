from sbert import SBert
from numpy import dot
from numpy.linalg import norm
from scipy import spatial
import csv



names=[]
labels=[]

with open('input.csv', 'r') as file:

# create the csv writer
    reader = csv.reader(file)
    model = SBert('paraphrase-MiniLM-L6-v2')
    i=0
    for row in reader:
        names.append(row[1])
        labels.append(row[0])
sentence_embeddings = model.encode(names)
with open('alp.csv','w') as f:
    fieldnames=['label','content']
    writer=csv.DictWriter(f,fieldnames=fieldnames)
    writer.writeheader()
    for i in range(0,len(names)):
        writer.writerow({'label':labels[i],'content':sentence_embeddings[i]})

with open('alptest.csv','w') as f:

    fieldnames=['label','content']
    writer=csv.DictWriter(f,fieldnames=fieldnames)
    writer.writeheader()
    for i in range(0,len(names)):
        writer.writerow({'label':labels[i],'content':names[i]})









#Print the embeddings
