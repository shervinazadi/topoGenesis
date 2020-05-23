# Copyright ©2020, written by Dr. Pirouz Nourian, Genesis AI Lab, Delft, the Netherlands, April-May 2020
'''
Copyright (c) <2020> <Dr.Pirouz Nourian>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import os
import marshal
import tempfile
#https://packaging.python.org/tutorials/packaging-projects/
# def Send23(data,folderName,fileName):
#     ##note to take out the imports before running this repeatedly!
#     #packages data into marshal binary files from Py2 to Py3
#     dbname=folderName+fileName
#     dbfile = open(dbname, 'wb') # Its important to use binary mode https://docs.python.org/3/library/functions.html#open
#     marshal.dump((str(type(data)),data), dbfile)   # source, destination
#     dbfile.close()
#     print('succeeded in marshaling the input dictionarry!')
#     dbname=','.join(str(ord(c)) for c in dbname)#str(os.path.normpath(dbname)).replace("\\","\\\\")#ospath.join(folderName,fileName)
#     return dbname
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
# def Receive32(ASCIIfileName):
#     #returns a data envelope conventionally filled with a tuple of (dataTypeDescription,data)
#     dbname=''.join(chr(int(iC.split(',')[0])) for iC in fileName.split())
#     print(dbname)
#     dbfile = open(dbname, 'rb') # Its important to use binary mode https://docs.python.org/3/library/functions.html#open
#     dataEnvelope=marshal.load(dbfile)   # source, destination
#     return dataEnvelope
def Receive32(fileName):
    #returns a data envelope conventionally filled with a tuple of (dataTypeDescription,data)
    #dbname=''.join(chr(int(iC.split(',')[0])) for iC in fileName.split())
    dbname=os.path.normpath(os.path.join(tempfile.gettempdir(),fileName))
    print(dbname)
    dbfile = open(dbname, 'rb') # Its important to use binary mode https://docs.python.org/3/library/functions.html#open
    dataEnvelope=marshal.load(dbfile)   # source, destination
    return dataEnvelope
# def Send32(data, fileNameASCIIstring):
#     #fileName=''.join(str(fileNameASCIIstring))[1:-1]
#     y=str(fileNameASCIIstring)[1:-1].split(',')
#     dbnameString=''.join(chr(int(i)) for i in y)
#     print(dbnameString)
#     dbfile = open(dbnameString, 'wb')
#     marshal.dump((str(type(data)),data),dbfile,2)#in python2 data types
#     dbfile.close()
#     print("successfully dumped the data in python2 format")
#     fileName=''.join(str(fileNameASCIIstring))[1:-1]#dbnameString
#     return fileName
def Send32(data):
    #fileName=''.join(str(fileNameASCIIstring))[1:-1]
    dbfile=tempfile.NamedTemporaryFile('wb', delete=False)
    open(dbfile.name, 'wb')
    marshal.dump((str(type(data)),data),dbfile,2)#in python2 data types
    dbfile.close()
    print("successfully dumped the data in python2 format")
    #fileName=''.join(str(fileNameASCIIstring))[1:-1]#dbnameString
    return os.path.basename(os.path.normpath(dbfile.name))
# def Receive23(dbnameASCII):
#     #decoding file name from ASCII codes
#     y=str(dbnameASCII)[1:-1].split(',')
#     dbname=''.join(chr(int(i)) for i in y)
#     dbname=os.path.normpath(dbname)
#     print(dbname)
#     #reading file from filePath
#     dbfile = open(dbname, 'rb')
#     envelope = marshal.load(dbfile)
#     dbfile.close()
#     print(type(envelope))
#     return envelope
def Receive23(fileName):
    #returns a data envelope conventionally filled with a tuple of (dataTypeDescription,data)
    #dbname=''.join(chr(int(iC.split(',')[0])) for iC in fileName.split())
    dbname=os.path.normpath(os.path.join(tempfile.gettempdir(),fileName))
    print(dbname)
    dbfile = open(dbname, 'rb') # Its important to use binary mode https://docs.python.org/3/library/functions.html#open
    dataEnvelope=marshal.load(dbfile)   # source, destination
    return dataEnvelope
def ASCIIPathEncoder(filePath):
    ##this is a function originally written for IronPython
    dbname=','.join(str(ord(c)) for c in filePath)#str(os.path.normpath(dbname)).replace("\\","\\\\")#ospath.join(folderName,fileName)
    return dbname
# def filePathASCIIEncoder(folderName,fileName):
#     ##this is a function originally written for IronPython
#     dbname=folderName+fileName
#     dbname=','.join(str(ord(c)) for c in dbname)#str(os.path.normpath(dbname)).replace("\\","\\\\")#ospath.join(folderName,fileName)
#     return dbname
def ASCIIPathDecoder(ASCIIPathString):
    # decodes a path consisting of ASCII codes into a normala path, must be updated using the os module
    y=str(ASCIIPathString)[1:-1].split(',')
    dbnameString=''.join(chr(int(i)) for i in y)
    return os.path.normpath(dbnameString)

dictest2={}#a test datum, a dictionary that has two chpaters
dictest2[0]=[0,1,2]
dictest2[1]=[0,1,3,4]
dictest3={}#a test datum, a dictionary that has three chapters
dictest3[0]=[0,1,3,6,10,99]
dictest3[1]=[1,5]
dictest3[2]=[0,4,5]
'''
#####################
#### test data
#####################

outFile=Send32(dict3)

if not inFile==None:
    dict2=Receive23(inFile)
    print(dict2)

dictest2={}
dictest2[0]=[0,1,2]
dictest2[1]=[0,1,3,4]

outFile=Send23(dict2)
if not inFile==None:
    dict3=Receive32(inFile)
    print(dict3)
'''
