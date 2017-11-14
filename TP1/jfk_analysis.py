# TP1

### Q1
def question1() :
    print("\n# Open file")
    with open('jfkrelease-2017-dce65d0ec70a54d5744de17d280f3ad2.csv') as f:
        read_data = f.readlines()
        for line in read_data:
            print(line)
    f.closed

    print("\n# Check number of fields")
    with open('jfkrelease-2017-dce65d0ec70a54d5744de17d280f3ad2.csv') as f:
        header = f.readline()
        firstLineFieldLength = header.split(";").__len__();
        read_data = f.readlines()
        identicalNumberOfLine = True
        for line in read_data:
            lineField = line.split(";")
            if(lineField.__len__()!=firstLineFieldLength):
                identicalNumberOfLine = False
        print("The number of fields is identical for all lines : %r" % identicalNumberOfLine)
    f.closed

### Q2
def question2() :
    print("\n# Compute the mean number of pages per document, plus the minimum/maximum of pages per document")
    with open('jfkrelease-2017-dce65d0ec70a54d5744de17d280f3ad2.csv') as f:
        header = f.readline()
        read_data = f.readlines()
        count = 0.0
        sum = 0.0
        nbOfPages = []
        for line in read_data:
            nbPages = line.split(";")[11]
            if nbPages.isdigit():
                sum += int(nbPages)
                count+=1
        mean = sum/count
        print("Mean : %f" % mean)
    f.closed

    with open('jfkrelease-2017-dce65d0ec70a54d5744de17d280f3ad2.csv') as f:
        header = f.readline()
        firstLine = f.readline().split(";")
        min = firstLine[11]

        read_data = f.readlines()

        for line in read_data:
            nbPages = line.split(";")[11]
            if nbPages.isdigit():
                if(int(nbPages)<min):
                    min = int(nbPages)
        print ("Min : %f" % min)
    f.closed

    with open('jfkrelease-2017-dce65d0ec70a54d5744de17d280f3ad2.csv') as f:
        header = f.readline()
        firstLine = header.split(";")
        while not firstLine[11].isdigit() :
            firstLine = f.readline().split(";")
        max = float(firstLine[11])

        read_data = f.readlines()

        for line in read_data:
            nbPages = line.split(";")[11]
            if nbPages.isdigit():
                if(int(nbPages)>max):
                    max = int(nbPages)
        print ("Max : %f" % max)
    f.closed

    print("\n# Search for missing file pages")
    with open('jfkrelease-2017-dce65d0ec70a54d5744de17d280f3ad2.csv') as f:
        header = f.readline()
        missingPages = 0

        read_data = f.readlines()

        for line in read_data :
            fields = line.split(";")
            nbPages = fields[11]
            if nbPages != "" or not nbPages.isdigit()  :
                missingPages += 1
        print("Number of docs with missing pages : %d." % missingPages)
    f.closed


### Q3
def question3() :
    print("\n# Number of types of documents")
    with open('jfkrelease-2017-dce65d0ec70a54d5744de17d280f3ad2.csv') as f:
        header = f.readline()
        types = dict()
        read_data = f.readlines()

        for line in read_data :
            docType = line.split(";")[6]
            if types.get(docType) :
                types[docType] = types[docType]+1
            else :
                types[docType] = 1
        print("There are %d types of document in the collection" % len(types))
    f.closed

    print("\n# Compute the number of documents per document type")
    with open('jfkrelease-2017-dce65d0ec70a54d5744de17d280f3ad2.csv') as f:
        header = f.readline()
        types = dict()
        read_data = f.readlines()

        for line in read_data :
            docType = line.split(";")[6]
            if types.get(docType) :
                types[docType] = types[docType]+1
            else :
                types[docType] = 1
        for key, value in types.items():
            print("%s : %d" % (key,value))
    f.closed

    print("\n# Number of involved agencies")
    with open('jfkrelease-2017-dce65d0ec70a54d5744de17d280f3ad2.csv') as f:
        header = f.readline()
        agencies = dict()
        read_data = f.readlines()

        for line in read_data :
            fields = line.split(";")
            docAgency = fields[4]
            if agencies.get(docType) :
                agencies[docAgency] = agencies[docAgency]+1
            else :
                agencies[docAgency] = 1
        print("%d agencies are involved" % len(agencies))
    f.closed

    print("\n# Number of document per agency")
    with open('jfkrelease-2017-dce65d0ec70a54d5744de17d280f3ad2.csv') as f:
        header = f.readline()
        agencies = dict()
        read_data = f.readlines()

        for line in read_data :
            fields = line.split(";")
            docAgency = fields[4]
            if agencies.get(docAgency) :
                agencies[docAgency] = agencies[docAgency]+1
            else :
                agencies[docAgency] = 1
        for key, value in agencies.items():
            print("%s : %d" % (key,value))
    f.closed

### Q4
def oldestDate(date1, date2):
    #Parsing date1
    year1 = date1.split("/")[2]
    month1 = date1.split("/")[1]
    day1 = date1.split("/")[0]

    # Parsing date2
    year2 = date2.split("/")[2]
    month2 = date2.split("/")[1]
    day2 = date2.split("/")[0]

    if year1 < year2 :
        return date1
    elif year1 > year2 :
        return date2
    else :
        if month1 < month2 :
            return date1
        elif month1 > month2 :
            return date2
        else :
            if day1 < day2 :
                return date1
            else :
                return date2

def newestDate(date1, date2):
    #Parsing date1
    year1 = date1.split("/")[2]
    month1 = date1.split("/")[1]
    day1 = date1.split("/")[0]

    # Parsing date2
    year2 = date2.split("/")[2]
    month2 = date2.split("/")[1]
    day2 = date2.split("/")[0]

    if year1 < year2 :
        return date2
    elif year1 > year2 :
        return date1
    else :
        if month1 < month2 :
            return date2
        elif month1 > month2 :
            return date1
        else :
            if day1 < day2 :
                return date2
            else :
                return date1

def question4() :
    print("\n# Oldest and the more recent documents ?")
    with open('jfkrelease-2017-dce65d0ec70a54d5744de17d280f3ad2.csv') as f:
        header = f.readline()

        #Getting Name and Date of the first document
        oldestDocName = f.readline().split(";")[0]
        oldestDocDate = f.readline().split(";")[5]

        newestDocName = oldestDocName
        newestDocDate = oldestDocDate

        read_data = f.readlines()

        for line in read_data:
            fields = line.split(";")
            docName = fields[0]
            docDate = fields[5]

            if docDate != "" and len(docDate.split("/")) == 3 and docDate.split("/")[2]!="0000":
                if oldestDate(docDate, oldestDocDate) == docDate :
                    oldestDocDate = docDate
                    oldestDocName = docName
                if newestDate(docDate, newestDocDate) == docDate :
                    newestDocDate = docDate
                    newestDocName = docName

        print("Oldest Document : %s (%s)" % (oldestDocName,oldestDocDate))
        print("Newest Document : %s (%s)" % (newestDocName,newestDocDate))

    f.closed

    print("\n# Number of documents per year")
    with open('jfkrelease-2017-dce65d0ec70a54d5744de17d280f3ad2.csv') as f:
        years = dict()
        read_data = f.readlines()

        for line in read_data :
            docDate = line.split(";")[5]
            if docDate != "" and len(docDate.split("/"))==3 :
                docYear = docDate.split("/")[2]
                if years.get(docYear) :
                    years[docYear] = years[docYear]+1
                else :
                    years[docYear] = 1
        for key, value in years.items():
            print("%s : %d" % (key,value))
    f.closed

