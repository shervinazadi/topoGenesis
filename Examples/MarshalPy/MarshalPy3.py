import os
import marshal
import tempfile
#https://packaging.python.org/tutorials/packaging-projects/
def Send32(data, fileNameASCIIstring):
    #fileName=''.join(str(fileNameASCIIstring))[1:-1]
    y=str(fileNameASCIIstring)[1:-1].split(',')
    dbnameString=''.join(chr(int(i)) for i in y)
    print(dbnameString)
    dbfile = open(dbnameString, 'wb')
    marshal.dump((str(type(data)),data),dbfile,2)#in python2 data types
    dbfile.close()
    print("successfully dumped the data in python2 format")
    fileName=''.join(str(fileNameASCIIstring))[1:-1]#dbnameString
    return fileName

def Send32(data):
    #fileName=''.join(str(fileNameASCIIstring))[1:-1]
    dbfile=tempfile.NamedTemporaryFile('wb', delete=False)
    open(dbfile.name, 'wb')
    marshal.dump((str(type(data)),data),dbfile,2)#in python2 data types
    dbfile.close()
    print("successfully dumped the data in python2 format")
    #fileName=''.join(str(fileNameASCIIstring))[1:-1]#dbnameString
    return os.path.basename(os.path.normpath(dbfile.name))

def Receive23(dbnameASCII):
    #decoding file name from ASCII codes
    y=str(dbnameASCII)[1:-1].split(',') 
    dbname=''.join(chr(int(i)) for i in y)
    dbname=os.path.normpath(dbname)
    print(dbname)
    #reading file from filePath
    dbfile = open(dbname, 'rb')      
    envelope = marshal.load(dbfile)
    dbfile.close()
    print(type(envelope))
    return envelope

def Receive23(fileName):
    #returns a data envelope conventionally filled with a tuple of (dataTypeDescription,data)
    #dbname=''.join(chr(int(iC.split(',')[0])) for iC in fileName.split())
    dbname=os.path.normpath(os.path.join(tempfile.gettempdir(),fileName))
    print(dbname)
    dbfile = open(dbname, 'rb') # Its important to use binary mode https://docs.python.org/3/library/functions.html#open
    dataEnvelope=marshal.load(dbfile)   # source, destination
    return dataEnvelope

dict3={}
dict3[0]=[0,1,3,6,10,99]
dict3[1]=[1,5]
dict3[2]=[0,4,5]
outFile=Send32(dict3)

if not inFile==None:
    dict2=Receive23(inFile)
    print(dict2)
