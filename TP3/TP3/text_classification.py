# TP3

## Q1

def question1() :
    print("\n# Open file")
    with open('LeMonde2003.csv') as f:
        read_data = f.readlines()
        for line in read_data:
            print(line)
    f.closed

df.dropna(axis=1, how='all')