import sys
import fractions
from math import ceil
def _lcm(a,b): return abs(a * b) / fractions.gcd(a,b) if a and b else 0
def lcm(a):
	return reduce(_lcm, a)

def generate_arrival_pattern(dags):
	arrival_instances=[]
	periods=[]
	for d in dags:
		periods.append(dags[d])
	hyperperiod=lcm(periods)
	for d in dags:
		i=0.0
		for i in xrange(0,int(ceil(hyperperiod/dags[d]))):
			release=dags[d]*i
			task_arrival_pair = (d,release)
			arrival_instances.append(task_arrival_pair)
	arrival_instances = sorted(arrival_instances,key=lambda x: x[1])
	return arrival_instances

def generate_arrival_file(arrival_instances,filename):
	f=open(filename,'w')
	for a in arrival_instances:
		line = a[0]+":"+str(a[1]*1e6)+"\n"
		f.write(line)



if __name__ == '__main__':
	dag_file=sys.argv[1]
	dags = {}
	dag_info=open(dag_file,'r').readlines()
	for line in dag_info:
		info=line.strip("\n").split(":")
		d=info[0]
		dags[d]=float(info[1])
	arrival_instances=generate_arrival_pattern(dags)
	for a in arrival_instances:
		print a
	generate_arrival_file(arrival_instances,sys.argv[2])
	