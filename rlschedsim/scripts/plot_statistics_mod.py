import numpy as np
import matplotlib.pyplot as plt
import sys
from collections import Counter
from engine import *
plt.rcParams.update({'font.size':10})

def calc_throughput_requirement(task_periods, flops_values):
	throughput_requirements = []
	for t in task_periods:
		tr = 0
		for i in range(4):
			if i%2 == 0:
				tr += (flops_values[0]/t[i])*1000
			else:
				tr += (flops_values[1]/t[i])*1000

		throughput_requirements.append([tr,t])

	return throughput_requirements

def generate_task_set_periods(period_configuration_file,ex_map):
	def combination_with_repetition(possible_periods_hyper_cycle,hyper_period,did,dag_period_map):
		if did == len(dag_period_map.keys())-1:
			for p in dag_period_map[did]:
				period = deepcopy(hyper_period)
				period[did] = p
				possible_periods_hyper_cycle.append(period)
		else:
			for p in dag_period_map[did]:
				period = deepcopy(hyper_period)
				period[did] = p
				combination_with_repetition(possible_periods_hyper_cycle, period, did+1, dag_period_map)

			   
	contents = open(period_configuration_file,'r').readlines()
	dag_period_map = {}
	dags = []
	counter_id = 0
	global_file_list = open("global_map.txt", "r").readlines()
	global_map = make_dict(global_file_list)
	DC = DAGCreator()
	for line in contents:
		dag_name,period_configuration = line.strip("\n").split("=")
		dag_period_map[counter_id] = eval(period_configuration)
		input_file = "ADAS_Graphs/"+dag_name
		dags.append(DC.create_adas_job_from_file_with_fused_times(input_file, counter_id, global_map, ex_map, counter_id,0.0))
		counter_id += 1
	possible_periods_hyper_cycle = []
	hyper_period = [0 for d in dag_period_map.keys()]
	combination_with_repetition(possible_periods_hyper_cycle,hyper_period,0,dag_period_map)
	return dags,possible_periods_hyper_cycle


def plot_bar_chart(baseline_period_stats,dag_period_stats,index,period_file,flops,filename):
	ex_cpu = {}
	ex_gpu = {}
	ex_map = (ex_cpu, ex_gpu)
	dags, task_periods = generate_task_set_periods(period_file,ex_map)
	task_periods_string = [str(t) for t in task_periods]
	baseline_stats = []
	rl_stats = []
	baseline_win = 0
	rl_win = 0
	draw = 0
	rl_max = float("-inf")
	baseline_max = float("-inf")
	baseline_win_periods = []
	counter = 0
	rl_pos_lateness = 0
	baseline_pos_lateness = 0
	for period in task_periods:
		baseline_value = baseline_period_stats[str(period)][index]
		rl_value = rl_period_stats[str(period)][index]
		rl_max = max(rl_max,rl_value)
		baseline_max = max(baseline_max,baseline_value)
		if rl_value <= 0.0 and baseline_value <0.0:
			if rl_value < baseline_value:
				rl_win += 1
			elif baseline_value < rl_value:
				baseline_win +=1
			else:
				draw +=1
		
		elif rl_value >=0.0 and baseline_value >=0.0:
			if index == 1 or index == 4:
				rl_pos_lateness +=1
				baseline_pos_lateness +=1
			if rl_value < baseline_value:
				rl_win +=1
			elif baseline_value < rl_value:
				baseline_win +=1
				baseline_win_periods.append((counter,period))
			else:
				draw += 1
		else:
			if rl_value >=0.0 and baseline_value <= 0.0:
				if index ==1 or index ==4:
					rl_pos_lateness +=1
				if rl_value != baseline_value:                    
					baseline_win +=1
					baseline_win_periods.append((counter,period))
				else:
					draw +=1
				
			elif rl_value <=0.0 and baseline_value >=0.0:
				if index == 1 or index ==4:
					baseline_pos_lateness +=1
				if rl_value != baseline_value:                                       
					rl_win += 1
				else:
					draw +=1
			
		baseline_stats.append(baseline_period_stats[str(period)][index])
		rl_stats.append(rl_period_stats[str(period)][index])
		
		counter +=1

	print baseline_stats
	print rl_stats

	fig = plt.figure(figsize=(18,6))
	ax = fig.add_subplot(1,1,1)

	# fig, ax = plt.subplots(1,figsize=(100,100))
	width = 1.2
	tr_vals = calc_throughput_requirement(task_periods, flops)
	temp = []
	for a,b,c in zip(tr_vals,baseline_stats,rl_stats):
		temp.append([a[0],b,c,a[1]])

	temp.sort()
	new_temp = []
	for i in temp:
		if i[0] > 2.8:
			new_temp.append(i)

	temp = new_temp
	temp.sort()
	print "Temp",len(temp)

	temp_tr = []
	tr_vals = []
	baseline_stats = []
	rl_stats = []
	for i in range(len(temp)):
		tr_vals.append([temp[i][0],temp[i][3]])
		temp_tr.append(temp[i][0])
		baseline_stats.append(temp[i][1])
		rl_stats.append(temp[i][2])

	print "Temp_tr",len(temp_tr)
	# print "throughput_requirements, "
	for i,j in tr_vals:
		print i,j
	start_indices = []
	prev = temp_tr[0]
	for i in range(1,len(temp_tr)):
		curr = temp_tr[i]
		if not prev == curr:
			start_indices.append(i-0.5)
		prev = curr

	print start_indices

	count_tr = Counter(temp_tr)
	print count_tr
	tr_set = set(temp_tr)
	tr_list = list(tr_set)
	tr_list.sort()
	# print tr_list

	tr_ind = {}
	cur_count = 0
	for tt in tr_list:
		tr_ind[tt] = cur_count + (count_tr[tt]+1)/2
		cur_count += count_tr[tt]

	print tr_ind

	tr_labels = [""]*(len(temp_tr))
	for ii in tr_ind:
		tr_labels[tr_ind[ii]] = str(ii)[:4]

	print len(baseline_stats)
	print len(rl_stats)

	ind = np.arange(0,2*len(tr_labels),2)
	# ind = np.arange(len(tr_labels))
	print ind
	ax.barh(ind, baseline_stats, width, color='red', label='Baseline')
	ax.barh([i+0.03 for i in ind] , rl_stats, 0.6, color='green', label='RL Agent')
	# tr_vals_str = []
	# for i in tr_vals:
	#     tr_vals_str.append(str(i[0])[:4])
	# ax.set(yticks=ind, yticklabels=tr_labels, ylim=[2*width - 1, len(tr_labels)])
	ax.set(yticks=ind, yticklabels=tr_labels)

	for val in start_indices:
		plt.axhline(y=2*val, color='grey', linestyle='--')
	plt.yticks(fontsize=30)
	plt.xticks(fontsize=30)
	ax.legend(loc='lower right', fontsize=22)

	if index == 1:
		plt.xlabel('Max Lateness', fontsize=30)
	else:
		plt.xlabel('Avg Lateness', fontsize=30)
	plt.ylabel('TI (e+12)', fontsize=30)
	plt.show()
	# for i in range(len(tr_vals_str)):
	#     if baseline_stats[i] > 800:
	#         print tr_vals[i][0], tr_vals[i][1]
	fig.savefig(filename, bbox_inches='tight')
	print "Baseline Wins",baseline_win
	print baseline_win_periods
	print "RL Wins",rl_win
	print "Number of draws",draw
	
	print "Baseline Max",baseline_max
	print "RL Max", rl_max


def plot_line_plots(baseline_period_stats,dag_period_stats,index,period_file,flops,filename):
	filename=filename[:-4]+"_line.pdf"
	ex_cpu = {}
	ex_gpu = {}
	ex_map = (ex_cpu, ex_gpu)
	dags, task_periods = generate_task_set_periods(period_file,ex_map)
	task_periods_string = [str(t) for t in task_periods]
	baseline_stats = []
	rl_stats = []
	baseline_win = 0
	rl_win = 0
	draw = 0
	rl_max = float("-inf")
	baseline_max = float("-inf")
	baseline_win_periods = []
	counter = 0
	rl_pos_lateness = 0
	baseline_pos_lateness = 0
	for period in task_periods:
		baseline_value = baseline_period_stats[str(period)][index]
		rl_value = rl_period_stats[str(period)][index]
		rl_max = max(rl_max,rl_value)
		baseline_max = max(baseline_max,baseline_value)
		if rl_value <= 0.0 and baseline_value <0.0:
			if rl_value < baseline_value:
				rl_win += 1
			elif baseline_value < rl_value:
				baseline_win +=1
			else:
				draw +=1
		
		elif rl_value >=0.0 and baseline_value >=0.0:
			if index == 1 or index == 4:
				rl_pos_lateness +=1
				baseline_pos_lateness +=1
			if rl_value < baseline_value:
				rl_win +=1
			elif baseline_value < rl_value:
				baseline_win +=1
				baseline_win_periods.append((counter,period))
			else:
				draw += 1
		else:
			if rl_value >=0.0 and baseline_value <= 0.0:
				if index ==1 or index ==4:
					rl_pos_lateness +=1
				if rl_value != baseline_value:                    
					baseline_win +=1
					baseline_win_periods.append((counter,period))
				else:
					draw +=1
				
			elif rl_value <=0.0 and baseline_value >=0.0:
				if index == 1 or index ==4:
					baseline_pos_lateness +=1
				if rl_value != baseline_value:                                       
					rl_win += 1
				else:
					draw +=1
			

		baseline_stats.append(baseline_period_stats[str(period)][index])
		rl_stats.append(rl_period_stats[str(period)][index])
		
		counter +=1

	print baseline_stats
	print rl_stats

	fig = plt.figure(figsize=(18,6))
	ax = fig.add_subplot(1,1,1)

	# fig, ax = plt.subplots(1,figsize=(100,100))
	width = 0.4
	tr_vals = calc_throughput_requirement(task_periods, flops)
	print "TR_VALS",tr_vals
	temp = []
	for i in range(len(tr_vals)):
		temp.append([tr_vals[i][0],baseline_stats[i],rl_stats[i],tr_vals[i][1],i])

	temp.sort()
	temp_tr = []
	temp_baseline_stats = {}
	for i in range(len(temp)):
		tr_vals[i] = [temp[i][0],temp[i][3],temp[i][4]]
		temp_tr.append(temp[i][0])
		baseline_stats[i] = temp[i][1]
		rl_stats[i] = temp[i][2]
		temp_baseline_stats[str(temp[i][3])] = [temp[i][1],temp[i][2]]

		print temp[i][0], temp[i][3], temp[i][1], temp[i][2]

	# for i,j in zip(tr_vals,rl_stats):
	# 	print i[0],i[2],j

	# for i in range(len(tr_vals)):
	# 	# print rl_stats[i]
	# 	if rl_stats[i] > 0.1 and rl_stats[i] < 0.20:
	# 		print rl_stats[i],tr_vals[i][1],tr_vals[i][2]

	ind = np.arange(len(task_periods))

	tr_set = set(temp_tr)
	tr_list = list(tr_set)
	tr_list.sort()
	print "tr_list",tr_list

	reduction_map = {}
	reverse_map = {}
	# reduction_map[tr_list[0]] = [tr_list[0]]
	# reverse_map[tr_list[0]] = tr_list[0]
	# prev = tr_list[0]
	# for i in range(1,len(tr_list)):
	# 	value = tr_list[i]/(1e9)
	# 	old_value = prev/(1e9)
	# 	if value - old_value <= 0.05:
	# 		reduction_map[prev].append(tr_list[i])
	# 		reverse_map[tr_list[i]] = prev
	# 	else:
	# 		reduction_map[tr_list[i]] = [tr_list[i]]
	# 		reverse_map[tr_list[i]] = tr_list[i]
	# 		prev = tr_list[i]

	reduction_map[7] = [tr_list[0]]
	reverse_map[tr_list[0]] = 7
	prev = 7
	for i in range(1,len(tr_list)):
		# value = tr_list[i]/(1e9)
		# old_value = prev/(1e9)
		# if value - old_value <= 0.05:
		# 	reduction_map[prev].append(tr_list[i])
		# 	reverse_map[tr_list[i]] = prev
		# else:
		# 	reduction_map[tr_list[i]] = [tr_list[i]]
		# 	reverse_map[tr_list[i]] = tr_list[i]
		# 	prev = tr_list[i]
		value = tr_list[i]/(1e9)
		print value
		print prev
		if value < 8:
			reduction_map[7].append(tr_list[i])
			reverse_map[tr_list[i]] = 7
		elif value >=8 and value < 9:
			if prev == 7:
				reduction_map[8] = [tr_list[i]]
				reverse_map[tr_list[i]] = 8
			else:
				reduction_map[8].append(tr_list[i])
				reverse_map[tr_list[i]] = 8
			prev = 8
		else:
			if prev == 8:
				reduction_map[9] = [tr_list[i]]
				reverse_map[tr_list[i]] = 9
			else:
				reduction_map[9].append(tr_list[i])
				reverse_map[tr_list[i]] = 9
			prev = 9

	print "reduction_map"

	for i in reduction_map:
		print i, len(reduction_map[i])

	new_temp_tr = []
	for i in temp_tr:
		new_temp_tr.append(reverse_map[i])

	print "***TR VALS and Task Periods***\n"
	tr_tp = []
	for i in tr_vals:
		tr_tp.append([reverse_map[i[0]],i[1]])

	tr_tp.sort()
	for i in tr_tp:
		print str(i[0]/(1e9))[:4], i[1], temp_baseline_stats[str(i[1])][0], temp_baseline_stats[str(i[1])][1]

	count_tr = Counter(new_temp_tr)
	print count_tr
	start_indices = []
	prev = new_temp_tr[0]
	for i in range(1,len(new_temp_tr)):
		curr = new_temp_tr[i]
		if not prev == curr:
			start_indices.append(i-0.5)
		prev = curr

	tr_set = set(new_temp_tr)
	tr_list = list(tr_set)
	tr_list.sort()
	print "tr_list",tr_list

	tr_ind = {}
	cur_count = 0
	for tt in tr_list:
		tr_ind[tt] = cur_count + (count_tr[tt]+1)/2
		cur_count += count_tr[tt]

	print tr_ind

	# tr_labels = [""]*(2*len(temp_tr))
	tr_labels = [""]*(len(ind))
	print tr_ind
	for ii in tr_ind:
		# label_val = str(float(str(ii)[:3])/100)
		# if len(label_val) > 0 and len(label_val) < 4:
		# 	label_val += "0"
		# label_val = "[" + str(ii) + "," + str(ii+1) + ")"
		label_val = "              " + str(ii)
		tr_labels[tr_ind[ii]-1] = label_val

	new_tr_labels = tr_labels
	labels_list = ["I","II","III"]
	label_counter = 0
	for i in range(len(tr_labels)):
		if len(tr_labels[i]) > 0:
			new_tr_labels[i] = tr_labels[i][:-1] + labels_list[label_counter]
			label_counter += 1

	tr_labels = new_tr_labels
	print "tr_labels",tr_labels

	baseline_stats_percentage = baseline_stats
	rl_stats_percentage = rl_stats

	print "baseline_stats", baseline_stats

	for i in range(len(rl_stats)):
		baseline_stats_percentage[i] = baseline_stats[i]*100
		rl_stats_percentage[i] = rl_stats[i]*100

	# ax.barh(ind, baseline_stats, width, color='red', label='B')
	# ax.barh(ind + width, rl_stats, width, color='green', label='RL')
	# ax.set(yticks=ind + width, yticklabels=task_periods_string, ylim=[2*width - 1, len(task_periods)])
	# ax.legend()
	plt.plot(ind,baseline_stats_percentage,'o',label='Baseline',linestyle='-',markersize=8)
	plt.plot(ind,rl_stats_percentage,'o',label='RL Agent',linestyle='-',markersize=8)
	for val in start_indices:
		plt.axvline(x=val, color='grey', linestyle='--')
	# tr_vals_str = []
	# for i in tr_vals:
	#     tr_vals_str.append(str(i)[:4] + " e12")
	# ax.set(yticks=ind, yticklabels=tr_vals_str, ylim=[2*width - 1, len(task_periods)])
	plt.xticks(ind,tr_labels,fontsize=15)
	plt.yticks(fontsize=30)
	plt.ylabel('Deadlines Missed (%)', fontsize=30)
	plt.xlabel('Configuration Clusters', fontsize=30)
	ax.legend(loc='upper left', fontsize=22)
	plt.show()
	fig.savefig(filename, bbox_inches='tight')
	# plt.savefig(filename, bbox_inches='tight')
	

if __name__ == '__main__':
	
	baseline_file = sys.argv[1]
	rl_file = sys.argv[2]
	statistic = sys.argv[3]
	period_file = sys.argv[4]
	filename = sys.argv[5]
	index_statistic_map = {"deadline_missed":0, "max_lateness":1, "tardiness":2, "avg_makespan":3, "avg_lateness":4 }

	#print(index_statistic_map)
	# flops_values = [4859136,546865296]
	flops_values = [9749885,556654736]	
	baseline_file_contents=open(baseline_file,'r').readlines()
	rl_file_contents=open(rl_file,'r').readlines()

	baseline_period_stats = {}
	rl_period_stats = {}

	for b in baseline_file_contents:
		period,stats = b.strip("\n").split(":")
		stats = map(float,stats.split(","))
		baseline_period_stats[period] = stats

	#print(baseline_period_stats)
	
	for rl in rl_file_contents:
		period,stats = rl.strip("\n").split(":")
		stats = map(float,stats.split(","))
		rl_period_stats[period] = stats
	
	ex_cpu = {}
	ex_gpu = {}
	ex_map = (ex_cpu, ex_gpu)

	# period_configurations = generate_task_set_periods(period_file)
	dags, period_configurations = generate_task_set_periods(period_file, ex_map)
	num_jobs = len(dags)
	
	for period in period_configurations:
		print period, baseline_period_stats[str(period)], rl_period_stats[str(period)]
	
	if index_statistic_map[statistic] == 0:
		plot_line_plots(baseline_period_stats,rl_period_stats,index_statistic_map[statistic],period_file,flops_values,filename)
	else:
		plot_bar_chart(baseline_period_stats,rl_period_stats,index_statistic_map[statistic],period_file,flops_values,filename)


