import os

line = open(os.path.join('/home/ubuntu/bjh/Gan/archive/list_attr_celeba.csv'),'r').readlines()
a = line[0].split(',')
lines = line[1:]
attr2idx = {}
idx2attr = {}
all_attr_name = self.lines[0].split(',')[1:]
for i, attr_name in enumerate(all_attr_name):
    attr2idx[attr_name] = i
    idx2attr[i] = attr_name
        
lines = self.lines[1:]
random.seed(2)
random.suffle(lines)
for i, line in enumerate(lines):
            split = line.split(',')
            split[-1] = split[-1].replace('\n','')
            filename = os.path.join(split[0])
            value = split[1:]

            label = []

            for attr_name in self.selected_attrs:
                idx = attr2idx[attr_name]

                if value[idx] == '1':
                    label.append(1)
                else:
                    label.append(0)
            print(labels)