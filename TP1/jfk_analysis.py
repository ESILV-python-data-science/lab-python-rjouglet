# TP1

# Question 1

# Open the CSV file and read all the lines
import time

columnNb = 0
sameColNb = True
avgPage = 0
sumPage = 0
maxPage = 0
minPage = 0
rows = 0
documents = 0
missingPageDoc = 0
docTypes = {}
agencies = {}
dates = {}
dateMax = time.strptime('04/20/1960', '%m/%d/%Y')
dateMin = time.strptime('04/20/1960', '%m/%d/%Y')

with open('jfkrelease.csv') as f:
    next(f)
    for row in f:
        rows = rows+1
        list = row.split(';')

        if columnNb == 0:
            columnNb = len(list)

        try:
            date = time.strptime(list[5], '%m/%d/%Y')
            if dateMax < date:
                dateMax = date
            if dateMin > date:
                dateMin = date
            if date not in dates:
                dates[date] = 1
            else:
                dates[date] = dates[date] + 1
        except ValueError:
            pass

        docType = list[6]
        if docType not in docTypes:
            docTypes[docType] = 1
        else:
            docTypes[docType] = docTypes[docType]+1
        agency = list[4]
        if agency not in agencies:
            agencies[agency] = 1
        else:
            agencies[agency] = agencies[agency]+1

        if list[11].isdigit():
            pageNumber = int(list[11])
            documents = documents +1
            sumPage = sumPage + pageNumber
            if maxPage < pageNumber:
                maxPage = pageNumber
            if minPage > pageNumber:
                minPage = pageNumber
        else:
            missingPageDoc = missingPageDoc+1
        if len(list) != columnNb:
            sameColNb = False;
    if sameColNb is True:
        print 'same amount of columns for each row'
    else:
        print 'not the same amount of columns for each row'
    avgPage = sumPage / documents
    print'average nb of page per doc = ',avgPage
    print 'max nb of page per doc = ',maxPage
    print 'min nb of page per doc = ',minPage
    print 'number of doc without pages indicated = ',missingPageDoc
    print 'the document with 0 pages is commented MISSING SERIAL'
    print 'number of different type of documents = ', len(docTypes) # some types are not defined (blank)
    for i in docTypes:
        print i, ' > ',docTypes[i]
    print 'number of different agencies = ', len(agencies) # some agencies are not defined (blank)
    for i in agencies:
        print i, ' > ',agencies[i]
    print 'here are the dates with day/month/year format'
    print 'date max = ',dateMax[2], '/', dateMax[1], '/', dateMax[0]
    print 'date min = ',dateMin[2], '/', dateMin[1], '/', dateMin[0]
    for i in dates:
        print i[2], '/', i[1], '/', i[0], ' > ',dates[i]