import csv
f= open('vectores_palabras.csv','rb')
reader=csv.reader(f)
row_count=sum(1 for row in f)
print row_count
f.close()