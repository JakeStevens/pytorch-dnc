import glob
import matplotlib.pyplot as plt
import numpy as np

def geo_mean(arr):
  a = np.array(arr)
  return a.prod()**(1.0/len(a))

def parse_timing_file(f):
  timing_info = {}
  with open(f, 'r') as timing_f:
    for line in timing_f:
      label, time = line.split(':')
      if label not in timing_info.keys():
        timing_info[label] = float(time)
      else:
        timing_info[label] += float(time)
  return timing_info

def graph_piechart(labels, data, title):
  fig = plt.figure(4, figsize=(5,5))
  ax1 = fig.add_subplot(211)
  ax1.set_title(title)
  ax1.axis('equal')
  pie = ax1.pie(data, startangle=0)
  ax2 = fig.add_subplot(212)
  ax2.axis('off')
  ax2.legend(pie[0], labels, loc='center')

  fig.tight_layout()
  plt.savefig('data/' + title.replace(' ','') + '.png', bbox_inches='tight')
  #plt.show()

def analyze_slot_sweep(benchmark):
  overview_labels = ['Controller Time', 'Memory Access Time', 'Output Time']
  memory_labels = ['Heads NN time', 'Key Similarity Time', 
                  'Content Weighting Time', 'Location Interpolation Time',
                  'Shift Weighting Time', 'Read Time', 'Write Time']
  # Get the data from the timing files
  for f in glob.glob('data/' + benchmark + '_*.timing'):
    timing_info = parse_timing_file(f)
    overview_data = []
    memory_data = []
    print f
    for label in overview_labels:
      overview_data.append(timing_info[label])
    memory_data = []
    for label in memory_labels:
      memory_data.append(timing_info[label])
    title = f.split('/')[-1].split('.')[0]
    #graph_piechart(overview_labels, overview_data, title)
    graph_piechart(memory_labels, memory_data, title)

def analyze_mann_read_write_ratio(benchmark):
  for f in glob.glob('data/' + benchmark + '_*timing'):
    timing_info = parse_timing_file(f)
    time_ratio = timing_info['Read Time']/timing_info['Write Time']
    operations_ratio = timing_info['Read Operations']/timing_info['Write Operations']
    print f
    print "Time Elapsed Ratio (Read:Write): " + str(time_ratio)
    print "Num. Operations Ratio (Read:Write): " + str(operations_ratio)

def analyze_geo_mean_of_all_timings():
  overview_labels = ['Controller Time', 'Memory Access Time', 'Output Time']
  memory_labels = ['Heads NN time', 'Key Similarity Time', 
                  'Content Weighting Time', 'Location Interpolation Time',
                  'Shift Weighting Time', 'Read Time', 'Write Time']
  overview_data = []
  memory_data = []
  # Get the data from the timing files
  for f in glob.glob('data/*.timing'):
    timing_info = parse_timing_file(f)
    tmp_overview_data = []
    for label in overview_labels:
      tmp_overview_data.append(timing_info[label])
    tmp_memory_data = []
    for label in memory_labels:
      tmp_memory_data.append(timing_info[label])
    overview_data.append(tmp_overview_data)
    memory_data.append(tmp_memory_data)
  # Get the Geo. Means
  overview_data_means = []
  for i in range(len(overview_data[0])):
    arr = []
    for j in range(len(overview_data)):
      arr.append(overview_data[j][i])
    overview_data_means.append(geo_mean(arr))
  memory_data_means = []
  for i in range(len(memory_data[0])):
    arr = []
    for j in range(len(memory_data)):
      arr.append(memory_data[j][i])
    memory_data_means.append(geo_mean(arr))
  
  graph_piechart(overview_labels, overview_data_means,'Geomean of Benchmarks')
  graph_piechart(memory_labels, memory_data_means, 'Geomean of Benchmarks')

print 'Analyzing timings'
analyze_slot_sweep('associative-ntm')
#analyze_mann_read_write_ratio('copy-ntm')
