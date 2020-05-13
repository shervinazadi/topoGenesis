import os
import marshal
import tempfile
#https://packaging.python.org/tutorials/packaging-projects/
def Send23(data,folderName,fileName):
    ##note to take out the imports before running this repeatedly!
    #packages data into marshal binary files from Py2 to Py3
    dbname=folderName+fileName
    dbfile = open(dbname, 'wb') # Its important to use binary mode https://docs.python.org/3/library/functions.html#open
    marshal.dump((str(type(data)),data), dbfile)   # source, destination                    
    dbfile.close() 
    print('succeeded in marshaling the input dictionarry!')
    dbname=','.join(str(ord(c)) for c in dbname)#str(os.path.normpath(dbname)).replace("\\","\\\\")#ospath.join(folderName,fileName)
    return dbname

def Send23(data):
    #fileName=''.join(str(fileNameASCIIstring))[1:-1]
    #packages data into marshal binary files from Py2 to Py3
    dbfile=tempfile.NamedTemporaryFile('wb', delete=False)
    dbfile=open(dbfile.name, 'wb')
    marshal.dump((str(type(data)),data),dbfile,2)#in python2 data types
    dbfile.close()
    print("successfully dumped the data in python2 format")
    #fileName=''.join(str(fileNameASCIIstring))[1:-1]#dbnameString
    return os.path.basename(os.path.normpath(dbfile.name))
def ASCIIPathDecoder(ASCIIPathString):
    y=str(ASCIIPathString)[1:-1].split(',')
    dbnameString=''.join(chr(int(i)) for i in y)
    return dbnameString
def Receive32(ASCIIfileName):
    #returns a data envelope conventionally filled with a tuple of (dataTypeDescription,data)
    dbname=''.join(chr(int(iC.split(',')[0])) for iC in fileName.split())
    print(dbname)
    dbfile = open(dbname, 'rb') # Its important to use binary mode https://docs.python.org/3/library/functions.html#open
    dataEnvelope=marshal.load(dbfile)   # source, destination
    return dataEnvelope
def Receive32(fileName):
    #returns a data envelope conventionally filled with a tuple of (dataTypeDescription,data)
    #dbname=''.join(chr(int(iC.split(',')[0])) for iC in fileName.split())
    dbname=os.path.normpath(os.path.join(tempfile.gettempdir(),fileName))
    print(dbname)
    dbfile = open(dbname, 'rb') # Its important to use binary mode https://docs.python.org/3/library/functions.html#open
    dataEnvelope=marshal.load(dbfile)   # source, destination
    return dataEnvelope

dict2={}
dict2[0]=[0,1,2]
dict2[1]=[0,1,3,4]

outFile=Send23(dict2)
if not inFile==None:
    dict3=Receive32(inFile)
    print(dict3)
