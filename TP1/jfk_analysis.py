# TP1

# Question 1

# Open the CSV file and read all the lines
f = open('jfkrelease-2017-dce65d0ec70a54d5744de17d280f3ad2.csv', 'r')

for line in f:
    print(line,end='')

# Check that the number of fields is identical for all lines

numberFields = len(f.readline().split(';'))
print(numberFields)
for line in f:
    if numberFields == len(line.split(';')) :
        numberFields = len(line.split(';'))
    else:
        print("Error in the CVS file")
print("The CSV file is correct")