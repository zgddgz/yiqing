alloverlap = 0
alldis = 0
alln_intersect  = 0
num = 736
fi = open('D:/Google Downloads/yiqing/evaluate/li.txt')
for i,line in enumerate(fi):
    line = line.strip('\n')
    line = line.split(',')
    alloverlap+=float(line[0])
    alldis+=float(line[1])
    alln_intersect+=float(line[2])
    if(i==num):
        break
print('li:',alloverlap/num, alldis/num, alln_intersect/num)

alloverlap = 0
alldis = 0
alln_intersect = 0
num = 736
fi = open('D:/Google Downloads/yiqing/evaluate/energy.txt')
for i, line in enumerate(fi):
    line = line.strip('\n')
    line = line.split(',')
    alloverlap += float(line[0])
    alldis += float(line[1])
    alln_intersect += float(line[2])
    if(i == num):
        break
print('energy:', alloverlap/num, alldis/num, alln_intersect/num)

alloverlap = 0
alldis = 0
alln_intersect = 0
num = 736
fi = open('D:/Google Downloads/yiqing/evaluate/li+.txt')
for i, line in enumerate(fi):
    line = line.strip('\n')
    line = line.split(',')
    alloverlap += float(line[0])
    alldis += float(line[1])
    alln_intersect += float(line[2])
    if(i == num):
        break
print('li+:', alloverlap/num, alldis/num, alln_intersect/num)
