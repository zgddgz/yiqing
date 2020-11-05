import math
def prodata(date):
    with open('C:/Users/lenovo/Documents/WeChat Files/wxid_ynlzus6rb66j21/FileStorage/File/2020-05/LeiJ.txt', 'r+') as title:
        hrtitle = title.read()
        L1 = hrtitle.split()
    LeiJ = []
    for i in range(len(L1)):
        L = []
        L.append(L1[i].split(','))
        LeiJ.append(math.log10(float(L[0][date])+1))
    with open('C:/Users/lenovo/Documents/WeChat Files/wxid_ynlzus6rb66j21/FileStorage/File/2020-05/XinZ.txt', 'r+') as title:
        hrtitle = title.read()
        L2 = hrtitle.split()
    XinZ = []
    for i in range(len(L2)):
        L = []
        L.append(L2[i].split(','))
        XinZ.append(math.log10(float(L[0][date])+1))
    with open('C:/Users/lenovo/Documents/WeChat Files/wxid_ynlzus6rb66j21/FileStorage/File/2020-05/SiW.txt', 'r+') as title:
        hrtitle = title.read()
        L3 = hrtitle.split()
    SiW = []
    for i in range(len(L3)):
        L = []
        L.append(L3[i].split(','))
        SiW.append(float(L[0][date]))
    # print(SiW)
    with open('C:/Users/lenovo/Documents/WeChat Files/wxid_ynlzus6rb66j21/FileStorage/File/2020-05/ZhiY.txt', 'r+') as title:
        hrtitle = title.read()
        L4 = hrtitle.split()
    ZhiY = []
    for i in range(len(L4)):
        L = []
        L.append(L4[i].split(','))
        ZhiY.append(float(L[0][date]))
    return LeiJ,XinZ,SiW,ZhiY
    # print(LeiJ, XinZ, SiW, ZhiY)
prodata(1)
