# # import os
# # path = 'C:/Users/lenovo/Desktop/new Test csv'
# # files = os.listdir(path)
# # files.sort(key=lambda x: int(x.split('.')[0]))
# # for file in files:
# #     i_str = str(file[0:-4])
# #     print(i_str)
# #     file_name = i_str + '.txt'
# #     f = open('C:/Users/lenovo/Desktop//new Test/new Test chongdie/'+file_name, 'w')
# #     f.close()

# # import os     #生成所有堆
# # import re
# # path = 'C:/Users/lenovo/Desktop/new Test/200 chongdie'
# # files = os.listdir(path)
# # files.sort(key=lambda x: int(x.split('.')[0]))
# # for file in files:
# #     f = open(path+'/'+file)
# #     lines = f.readlines()
# #     for line in lines[1:]:
# #         line = line.strip('\n')
# #         label = re.findall(r'[A-Za-z]+[0-9]',line)
# #         print(label)
# #         with open('C:/Users/lenovo/Desktop/new Test/200 all chongdie.txt', 'a')as fi:
# #             fi.write(file[0:-4]+str(label)+'\n')

# # import re     #对所有堆排序
# # f = open('C:/Users/lenovo/Desktop/2.0/Test all chong - 副本.txt')
# # lines = f.readlines()
# # pro = []
# # allline = []
# # label = []
# # for line in lines:
# #     line = line.strip('\n')
# #     pro.append(line)
# #     pro.sort(key=lambda i: float(re.match(r'\d+\.\d',i).group()),reverse=True)
# # print(pro)
# # for i in range(len(pro)):
# #     with open('C:/Users/lenovo/Desktop/2.0/Test all chong sort.txt', 'a')as fi:
# #         fi.write(pro[i]+'\n')

# #     allline.append((re.findall(r'\d+\.\d*',line),re.findall(r'[A-Za-z]+[0-9]', line),re.findall(r',+\d+',line)))
# #     label.append(re.findall(r'[A-Za-z]+[0-9]', line))
# # print(allline)
# # print(label)

# # d_list = label
# # # print(d_list)
# # num1 = []
# # for dict_item in allline:
# #     val_a = dict_item[1]
# #     new_tuple = (val_a)
# #     if new_tuple in d_list:
# #         # d_list.append(new_tuple)
# #         num1.append(dict_item)
# # print(num1)

# # max = allline[0][1]
# # l = []
# # for i in range(len(allline)-1):
# #     if allline[i][1] == allline[i+1][1]:
# #         l.append(allline[])

# # # pro.sort(key=lambda i: float(re.match(r'\d+\.\d',i).group()),reverse=True)
# # print(pro)
# # for i in range(len(pro)):
# #     with open('C:/Users/lenovo/Desktop/Test all chong/Test all chong sort.txt', 'a')as fi:
# #         fi.write(pro[i]+'\n')



# # path = 'C:/Users/lenovo/Desktop/all chong/all chong.txt'
# # for line in open(path):
# #     line = line.strip('\n')
# #     print(line[1:-5])
# #     li = line[1:-5].strip(',').split(',')
# #     print(li)
# #     print(len(li))
    
# # s = ['1.dat', '10.dat', '5.dat']
# # s.sort(key=lambda i: int(re.match(r'\d+', i).group()))
# # print(s)

# # dict_list = [{'a': 1, 'b': 2, 'c': 1}, {'a': 1, 'b': 2, 'c': 8},
# #              {'a': 1, 'b': 2, 'c': 2}, {'a': 3, 'b': 5, 'c': 8}]
# # d_list = []
# # dict_list2 = []
# # l= []
# # for dict_item in dict_list:
# #     val_a = dict_item['a']
# #      new_tuple = (val_a)
# #     if new_tuple not in d_list:
# #         d_list.append(new_tuple)
# #         dict_list2.append(dict_item)
# #     else:
# #         print('被移除元素：', dict_item)
# #         l.append(dict_item)


# # print(l)


# import os  # 生成Test
# import re
# path = 'C:/Users/lenovo/Desktop/Test province/chongdie'
# files = os.listdir(path)
# files.sort(key=lambda x: int(x.split('.')[0]))
# for i in range(1,722):
#     for file in files:
#         f = open(path+'/'+file)
#         lines = f.readlines()
#         for line in lines[1:]:
#             line = line.strip('\n')
#             # label = re.findall(r'[\u4E00-\u9FA5]+', line)
#             # print(label)
#         #     with open('C:/Users/lenovo/Desktop/Test province/Test.txt', 'a')as fi:
#         #         fi.write(line[3:])
#         # with open('C:/Users/lenovo/Desktop/Test province/Test.txt', 'a')as fi:
#         #     fi.write(file[0:-4]+'\n')


# def cosVector(x, y):
#     if(len(x) != len(y)):
#         print('error input,x and y is not in the same space')
#         return
#     result1 = 0.0
#     result2 = 0.0
#     result3 = 0.0
#     for i in range(len(x)):
#         result1 += x[i]*y[i]  # sum(X*Y)
#         result2 += x[i]**2  # sum(X*X)
#         result3 += y[i]**2  # sum(Y*Y)
#     #print(result1)
#     #print(result2)
#     #print(result3)
#     print("result is "+str(result1/((result2*result3)**0.5)))  # 结果显示


# cosVector([2, 1], [1, 1])


# for i in range(1,1791):
#     print(i)
#     with open('C:/Users/lenovo/Desktop/Test province/v.txt', 'a')as fi:
#         fi.write(str(i)+'\n')

# import math
# print(pow(math.e,math.cos(0)))
    #    hideindex = []
    #    for i in range(len(hidepro)):
    #         hideindex.append(text_strings.index(hidepro[i]))
    #     print(hideindex)
    #     for i in range(len(hideindex)):
    #         text_strings = text_strings.remove(text_strings[hideindex[i]])
    #         anchors = anchors.remove(anchors[hideindex[i]])
    #         v = v.remove(v[hideindex[i]])


# a = [0, 1, 2, 7, 8, 5, 6]
# index = [3, 4]
# newa = []
# for i in range(len(a)):
#     if i not in index:
#         newa.append(a[i])
         
# print(newa)


# import re
# hidepro = []
# fi = open('C:/Users/lenovo/Desktop/Test province/score.txt')  # hide label
# for i, lin in enumerate(fi):  # x,y
#     lin = lin.strip('\n')
#     label = re.findall(r'\d+', lin)
#     hidepro.append(label)
# print(hidepro[0])
# newhidepro = []
# for i in range(len(hidepro[0])):
#     newhidepro.append(int(hidepro[0][i]))
# print(newhidepro[0]+1)

for i in range(3700,4000):
    with open('C:/Users/lenovo/Desktop/Test province/all Test.txt','a')as fi:
        fi.write(str(i)+'\n')

