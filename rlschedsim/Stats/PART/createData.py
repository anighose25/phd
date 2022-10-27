import os
import sys
f1name=sys.argv[1]

f1=open(f1name,"w")

for root, dirs, files in os.walk("/home/anirban/Experiment/SchedML/PART/"):
    for file in files:
        if file.endswith(".stats"):

            filename=str(os.path.join(root,file))
            print filename
            f=open(filename,"r")
            row=""
            file_contents=f.readlines()
	    features=file_contents[0:15]
	    print features
	    partition=file_contents[16:]
	    for line in features:
               words=line.split(" ")
               word=words[1]
               word=word.strip("\n")
               row+=word+","
                
            row=row[:-1]
            part=""
            print row
		
            f1.write(row+"\n")	

