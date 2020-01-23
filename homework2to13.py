from PIL import Image
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import *
import sys
import cv2
import numpy as np  
import matplotlib.pyplot as plt
import copy
import json
import pywt
import six


name = ""
lastname = ""

class leafNode(object):
    def __init__(self,value,frequency,):
        super(leafNode,self).__init__()
        self.value = value
        self.weight =frequency
    def isleaf(self):
        return True
    def getWeight(self):
        return self.weight
    def getValue(self):
        return self.value
class connectNode(object):
    def __init__(self, l_node,r_node):
        super(connectNode,self).__init__()
        self.weight = l_node.getWeight()+r_node.getWeight()
        self.lChild = l_node
        self.rChild = r_node
    def isleaf(self):
        return False
    def getWeight(self):
        return self.weight
    def getLeftNode(self):
        return self.lChild
    def getRightNode(self):
        return self.rChild
    

class HuffT(object):

    def __init__(self,flag,value = 0,frequency = 0,l_node = None,r_node = None):
        super(HuffT,self).__init__()
        if flag == 0:
            self.root = leafNode(value,frequency)
        else :
            self.root = connectNode(l_node.getRoot(),r_node.getRoot())
    def getRoot(self):
        return self.root
    def getWeight(self):
        return self.root.getWeight()
    def codeTree(self,root,code,fre):
        if root.isleaf():
            fre[root.getValue()] = code
            return None
        else :
            self.codeTree(root.getLeftNode(),code+'0',fre)
            self.codeTree(root.getRightNode(),code+'1',fre)
def buildTree(listHufftree):
    while len(listHufftree)>1:
        listHufftree.sort(key= lambda x:x.getWeight())
        temp1 = listHufftree[0]
        temp2 = listHufftree[1]
        listHufftree = listHufftree[2:]
        newNode = HuffT(1,0,0,temp1,temp2)
        listHufftree.append(newNode)
    return listHufftree[0]
class Point(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    @property
    def getX(self):
        return self.x
    def getY(self):
        return self.y
def printHuffTree(tree):
    print(tree.getValue(),tree.get)
class mainwindow(QMainWindow):
    def __init__(self):
        super().__init__()
        #self.resize(1000,600)
        self.initUI()
    def initUI(self):
        
        openAct = QAction('&打开',self)
        openAct.setShortcut('Ctrl+N')
        openAct.setStatusTip("打开图片")
        openAct.triggered.connect(self.openfile)

        lh2Act = QAction('&量化level2',self)
        lh2Act.triggered.connect(self.lh2)

        lh4Act = QAction('&量化level4',self)
        lh4Act.triggered.connect(self.lh4)

        lh8Act = QAction('&量化level8',self)
        lh8Act.triggered.connect(self.lh8)

        cyAct128 = QAction('&采样128x128',self)
        cyAct128.triggered.connect(self.cy128)

        cyAct64 = QAction('&采样64x64',self)
        cyAct64.triggered.connect(self.cy64)

        cyAct32 = QAction('&采样32x32',self)
        cyAct32.triggered.connect(self.cy32)
        
        wAct = QAction('位平面',self)
        wAct.triggered.connect(self.w)

        txtAct = QAction('bmp2txt',self)
        txtAct.triggered.connect(self.bmp2txt)

        histogramact = QAction('灰度直方图(左)',self)
        histogramact.triggered.connect(self.hisact1)
        histogramact1 = QAction('灰度直方图(右)',self)
        histogramact1.triggered.connect(self.hisact2)

        cutact = QAction("&阈值化",self)
        cutact.triggered.connect(self.cut)

        pointope1 = QAction("&线性点运算1",self)
        pointope1.triggered.connect(self.point1)

        pointope2 = QAction("&线性点运算2",self)
        pointope2.triggered.connect(self.point2)

        pointope3 = QAction("&非线性点运算1",self)
        pointope3.triggered.connect(self.point3)

        pointope4 = QAction("&非线性点运算2",self)
        pointope4.triggered.connect(self.point4)

        equallist = QAction("&灰度均衡",self)
        equallist.triggered.connect(self.equal)

        equallists = QAction("&灰度均衡(优化)",self)
        equallists.triggered.connect(self.equals)
        

        geo1 = QAction("&放大(最近邻插值)",self)
        geo1.triggered.connect(self.geoone)

        geo2 = QAction("&缩小(最近邻插值)",self)
        geo2.triggered.connect(self.geotwo)

        geo3 = QAction("&放大(双线性插值)",self)
        geo3.triggered.connect(self.geothree)

        geo4 = QAction("&缩小(双线性插值)",self)
        geo4.triggered.connect(self.geofouth)

        geo5 = QAction("&图像顺时针旋转(最近邻)",self)
        geo5.triggered.connect(self.geofifth)

        geo6 = QAction("&图像顺时针旋转(双线性)",self)
        geo6.triggered.connect(self.geosixth)

        

        geo8 =QAction("&图像平移(向上,最近邻)",self)
        geo8.triggered.connect(self.geoeigth)

        geo9 =QAction("&图像平移(向下,最近邻)",self)
        geo9.triggered.connect(self.geoninth)

        geo10 =QAction("&图像平移(向左,双线性)",self)
        geo10.triggered.connect(self.geotenth)

        geo11 =QAction("&图像平移(向右,双线性)",self)
        geo11.triggered.connect(self.geoeleventh)

        geo12 = QAction("&A图到B图",self)
        geo12.triggered.connect(self.geotwelfth)

        imgtrans1 = QAction("&傅里叶变换(高通滤波)",self)
        imgtrans1.triggered.connect(self.imgt1)

        imgtrans2 = QAction("&傅里叶变换(低通滤波)",self)
        imgtrans2.triggered.connect(self.imgt2)

        imgtrans5 = QAction("&傅里叶变换(带通滤波)",self)
        imgtrans5.triggered.connect(self.imgt5)

        imgtrans3 = QAction("&离散余弦变换(高通滤波)",self)
        imgtrans3.triggered.connect(self.imgt3)

        imgtrans4 = QAction("&离散余弦变换(低通滤波)",self)
        imgtrans4.triggered.connect(self.imgt4)

        imgtrans6 = QAction("&离散余弦变换(带通滤波)",self)
        imgtrans6.triggered.connect(self.imgt6)

        imgtrans7 = QAction("&小波变换",self)
        imgtrans7.triggered.connect(self.imgt7)

        colorprocess1 = QAction("&灰度化",self)
        colorprocess1.triggered.connect(self.cp1)

        colorprocess2 = QAction("&色彩转换",self)
        colorprocess2.triggered.connect(self.cp2)

        imgstren1 = QAction("&图像平滑(均值)",self)
        imgstren1.triggered.connect(self.is1)
        imgstren2 = QAction("&图像平滑(中值)",self)
        imgstren2.triggered.connect(self.is2)
        imgstren3 = QAction("&图像平滑(K邻域)",self)
        imgstren3.triggered.connect(self.is3)
        imgstren4 = QAction("&图像锐化(Roberts)",self)
        imgstren4.triggered.connect(self.is4)
        imgstren5 = QAction("&图像锐化(Sobel)",self)
        imgstren5.triggered.connect(self.is5)
        imgstren6 = QAction("&图像锐化(Prewitt)",self)
        imgstren6.triggered.connect(self.is6)
        imgstren7 = QAction("&任意模板",self)
        imgstren7.triggered.connect(self.is7)
        imgstren8 = QAction("&同态滤波",self)
        imgstren8.triggered.connect(self.is8)

        cutimg1 = QAction("&图像分割(Sobel)",self)
        cutimg1.triggered.connect(self.ct1)
        cutimg2 = QAction("&图像分割(Prewitt)",self)
        cutimg2.triggered.connect(self.ct2)
        cutimg3 = QAction("&图像分割(拉普拉斯)",self)
        cutimg3.triggered.connect(self.ct3)
        cutimg4 = QAction("&霍夫变换",self)
        cutimg4.triggered.connect(self.ct4)

        comimg1 = QAction("&Huffman编码",self)
        comimg1.triggered.connect(self.ci1)       
        comimg2 = QAction("&Huffman编码解码",self)
        comimg2.triggered.connect(self.ci2)   
         
        comimg4 = QAction("&Huffman图形1",self)
        comimg4.triggered.connect(self.ci4)   
        comimg8 = QAction("&Huffman图形2",self)
        comimg8.triggered.connect(self.ci8) 
        comimg5 = QAction("&游程编码",self)
        comimg5.triggered.connect(self.ci5) 
        comimg6 = QAction("&游程编码解码",self)
        comimg6.triggered.connect(self.ci6)  
        comimg7 = QAction("&算数编码",self)
        comimg7.triggered.connect(self.ci7) 




        menubar = self.menuBar()
        fileMenu = menubar.addMenu("&文件")
        fileMenu.addAction(openAct)
       

        lhMenu = menubar.addMenu("&量化")
        lhMenu.addAction(lh8Act)
        lhMenu.addAction(lh4Act)
        lhMenu.addAction(lh2Act)

        cyMenu = menubar.addMenu("&采样")
        cyMenu.addAction(cyAct128)
        cyMenu.addAction(cyAct64)
        cyMenu.addAction(cyAct32)

        wMenu = menubar.addMenu("&位平面")
        wMenu.addAction(wAct)

        txtMenu = menubar.addMenu("&bmp2txt")
        txtMenu.addAction(txtAct)

        histogram = menubar.addMenu("&灰度直方图")
        histogram.addAction(histogramact)
        histogram.addAction(histogramact1)
        histogram.addAction(cutact)

        pointope = menubar.addMenu("&点运算")
        pointope.addAction(pointope1)
        pointope.addAction(pointope2)
        pointope.addAction(pointope3)
        pointope.addAction(pointope4)
        pointope.addAction(equallist)
        pointope.addAction(equallists)

        geoany = menubar.addMenu("&几何运算")
        geoany.addAction(geo1)
        geoany.addAction(geo2)
        geoany.addAction(geo3)
        geoany.addAction(geo4)
        geoany.addAction(geo5)
        geoany.addAction(geo6)
        geoany.addAction(geo8)
        geoany.addAction(geo9)
        geoany.addAction(geo10)
        geoany.addAction(geo11)
        geoany.addAction(geo12)

        imgtrans = menubar.addMenu("&图像变换")
        imgtrans.addAction(imgtrans1)
        imgtrans.addAction(imgtrans2)
        imgtrans.addAction(imgtrans5)
        imgtrans.addAction(imgtrans3)
        imgtrans.addAction(imgtrans4)
        imgtrans.addAction(imgtrans6)
        imgtrans.addAction(imgtrans7)

        imgstren = menubar.addMenu("&图像增强")
        imgstren.addAction(imgstren1)
        imgstren.addAction(imgstren2)
        imgstren.addAction(imgstren3)
        imgstren.addAction(imgstren4)
        imgstren.addAction(imgstren5)
        imgstren.addAction(imgstren6)
        imgstren.addAction(imgstren7)
        imgstren.addAction(imgstren8)



        colorprocess = menubar.addMenu("&彩色图像处理")
        colorprocess.addAction(colorprocess1)
        colorprocess.addAction(colorprocess2)

        cutimg = menubar.addMenu("&图像分割")
        cutimg.addAction(cutimg1)
        cutimg.addAction(cutimg2)
        cutimg.addAction(cutimg3)
        cutimg.addAction(cutimg4)

        comimg = menubar.addMenu("&图像压缩")
        comimg.addAction(comimg1)
        comimg.addAction(comimg2)
   
        comimg.addAction(comimg4)
        comimg.addAction(comimg8)
        comimg.addAction(comimg5)
        comimg.addAction(comimg6)
        comimg.addAction(comimg7)
        

        self.label = QLabel(self)
        self.label.setFixedSize(700, 1000)
        self.label.move(60, 100)
        self.label.setStyleSheet("QLabel{background:white;}")

        self.label2 = QLabel(self)
        self.label2.setFixedSize(700, 1000)
        self.label2.move(840, 100)
        self.label2.setStyleSheet("QLabel{background:white;}")

       
        self.setGeometry(100,100,1600,1200)
        self.setWindowTitle("图像处理")
        self.show()
    def openfile(self):
        
        imgName,imgType = QFileDialog.getOpenFileName(self,"","")
        global name
        
        for index, ch in enumerate(imgName):
            if ch =="/":
                num = index
        name = './'+imgName[num+1:]
        
        if name :
            self.label.setPixmap(QPixmap(name))
    def hisact(self,name):
      
        img = cv2.imread(name,flags=0)

        height = img.shape[0]
        width = img.shape[1]

        graylist = np.zeros(256,np.uint64)
        count = 0
        
        for i in range(height):
        
            for j in range(width):
                graylist[img[i,j]]+=1
        
        for i in range(256):
            count = count + i * graylist[i]
        img = img.reshape([height*width,])  
        count2 = 0
        for i in range (256):
            if count2<int(height * width /2) and (count2+graylist[i])>=int(height * width /2):
                midnum = i
            count2 += graylist[i]
        count=count /(height*width)
        stddeviation = 0
       
        for i in range(256):
            
            stddeviation=stddeviation+ (i- count)**2*graylist[i]
        stddeviation = stddeviation/(height*width-1)
        stddeviation = stddeviation**0.5
                
        
        
        #展示大小
        plt.figure(figsize=(12,6))
        plt.plot(graylist)
        maxy = np.max(graylist)
        plt.axis([0,255,0,maxy+10])

        plt.text(256,0,"sum of pixels\n"+str(height*width)+"\n"+"average gray level\n"+str(count)+
        '\n'+"median gray\n"+str(midnum)+'\n'+"standard deviation\n"+str(stddeviation))
     
        
        plt.show()
    def ci8(self):
        global name
        f = open(name,'rb')
        filedata = f.read()
        filesize = f.tell()

        node = {}

        for i in range(filesize):
                pointnum = filedata[i]
                if pointnum in node.keys():
                    node[pointnum]+=1
                else:
                    node[pointnum] = 1

        hufftree = []


        for x in node.keys():
            tem = HuffT(0,x,node[x],None,None)
            hufftree.append(tem)
        length = len(node.keys())

        finalTree = buildTree(hufftree)
        finalTree.codeTree(finalTree.getRoot(),'',node)
        node1 = []
        for x in node.keys():
            node1.append([x,node[x]])
        node2=sorted(node1,key = lambda x:x[1])
        for i in range(len(node2)):
            j = node2[i][1]
            for t in j:
                if t=='0':
                    print("*--",end="")
                else:
                    print("^--",end="")
            print(node2[i][0])
    def ci7(self):
        f = open(name,'rb')
        filedata = f.read()
        filesize = f.tell()

        node = {}

        for i in range(filesize):
            pointnum = filedata[i]
            if pointnum in node.keys():
                node[pointnum]+=1
            else:
                node[pointnum] = 1
        for i in node.keys():
            node[i] = float(node[i])/float(filesize)
        t = sorted(node.items(),key =lambda v:v[1],reverse=True)


        for i in range(len(t)):
            temp = t[i]
            t[i] = list(temp)
        for i in range(1,len(t)):
            t[i][1] = t[i][1]+t[i-1][1]
        high = t[0][1]
        low = 0
        for i in  range(1,len(t)):
            low = low+(high-low)*t[i-1][1]
            high = low+(high-low)*t[i][1]
        
        print('(',low,',',high,')')
        code = ''
        media = (high+low)/2
        count = 0
        while media!=0:
            if media <1:
                code+='0'
            else :
                code +='1'
                media -=1
            media = media*2
            count+=1
            
        print(code)


    def ci6(self):
        f = open('tenth_5','rb')
        filedata = f.read()
        filesize = f.tell()
        i = 0
        decodeStr = []
   
        for i in range(int(filesize/2)):
            value = filedata[2*i]
            tempcount = filedata[2*i+1]
            decodeStr.append(value)
            decodeStr.append(tempcount)
        global lastname
        lastname = 'tenth_6.bmp'
        output = open(lastname,'wb')
        code = []
        i = 0
        while i <len(decodeStr):
            count = decodeStr[i]
            i+=1
            value = decodeStr[i]
            i+=1
            for j in range(count):
                code.append(value)
        for i in code:
            output.write(six.int2byte(i))
        output.close()
        
        if lastname :
            self.label2.setPixmap(QPixmap(lastname))

    def ci5(self):
        encode = []

        count = 0
        global name
        f = open(name,'rb')
        filedata = f.read()
        filesize = f.tell()
        last = filedata[0]
        output = open('tenth_5','wb')
        for i in range(filesize):
    # 遍历每个字符，统计它连续出现的次数
            if filedata[i] == last:
                count += 1
            else:
                encode.append(count)
                encode.append(last)
                count = 1
                last = filedata[i]
        encode.append(count)
        encode.append(last)
        for i in encode:
            output.write(six.int2byte(i))   
        output.close()

    def ci4(self):
        number,ok=QInputDialog.getInt(self,'查看','像素值')
        if ok:
            global name
            f = open(name,'rb')
            filedata = f.read()
            filesize = f.tell()


            node = {}

            for i in range(filesize):
                    pointnum = filedata[i]
                    if pointnum in node.keys():
                        node[pointnum]+=1
                    else:
                        node[pointnum] = 1

            hufftree = []
        # for tem in node.keys():
        #     print(tem,' : ',node[tem])

            for x in node.keys():
                tem = HuffT(0,x,node[x],None,None)
                hufftree.append(tem)
            length = len(node.keys())

            finalTree = buildTree(hufftree)
            finalTree.codeTree(finalTree.getRoot(),'',node)
            # print(node)
            newimg = np.zeros((512,512),np.uint8)
            newimg1 = np.zeros((512,512),np.uint8)
            newimg[0,256] = 128
            newimg[0,255] =255
            newimg[1,256] =255
            newimg[1, 255] =255
            newimg1[0,256] = 128
            newimg1[0,255] =255
            newimg1[1,256] =255
            newimg1[1, 255] =255
            for x in node.keys():
                c = node[x]
                color = 255
               
                h = 0
                lentemp = len(c)
                w = 256
                for i in range(lentemp):
                    if c[0] == '1':
                        for j in range(0,20,2):
                            newimg[h+j,w+j] = color   
                        h +=20
                        w =w+20
                    else :
                        for j in range(0,20,2):
                            newimg[h+j,w-j] = color
                        h +=20
                        w =w-20
                    c = c[1:]
                newimg[h,w] =color
                newimg[h-1,w-1] =color
                newimg[h,w-1] =color
                newimg[h-1,w] =color
            c = node[number]
            color = 255
               
            h = 0
            lentemp = len(c)
            w = 256
            for i in range(lentemp):
                if c[0] == '1':
                    for j in range(0,20,2):
                        newimg1[h+j,w+j] = color   
                    h +=20
                    w =w+20
                    newimg1[h,w-1] = color
                else :
                    for j in range(0,20,2):
                        newimg1[h+j,w-j] = color
                    h +=20
                    w =w-20
                    newimg1[h,w-1] = color
                c = c[1:]
            newimg1[h,w] =color
            newimg1[h-1,w-1] =color
            newimg1[h,w-1] =color
            newimg1[h-1,w] =color
            cv2.imshow('tenth_4',newimg)
            cv2.imshow('tenth_5',newimg1)
   

    def ci2(self):
        f = open('tenth_1','rb')
        filedata = f.read()
        filesize = f.tell()
        lentemp = 0
        for i in range(4):
            aa = filedata[i]
            lentemp = lentemp|aa
            if i <3:
                lentemp = lentemp<<8
        leafSize = lentemp
        rebuildHuffT={}
        for i in range(lentemp):
            value = filedata[4+5*i]
            tempcount = 0
            for j in range(1,5):
                aa = filedata[4+5*i+j]
                tempcount = tempcount|aa
                if j <4:
                    tempcount = tempcount<<8
            rebuildHuffT[value] = tempcount
        rebuildlist = []
        for x in rebuildHuffT.keys():
            tem = HuffT(0,x,rebuildHuffT[x],None,None)
            rebuildlist.append(tem)
        tem = buildTree(rebuildlist)
        tem.codeTree(tem.getRoot(),'',rebuildHuffT)
        output = open('tenth_2.bmp','wb')
        code = ''
        rootnode = tem.getRoot()
        for x in range(leafSize*5+4,filesize):
            c = filedata[x]
            for i in range(8):
                if c&128:
                    code = code +'1'
                else:
                    code = code +'0'
                c =c<<1 
            while len(code) >24:
                if rootnode.isleaf():
                    tem_byte = six.int2byte(rootnode.getValue())
                    output.write(tem_byte)
                    rootnode = tem.getRoot() 
                if code[0] == '1':
                    rootnode = rootnode.getRightNode()
                else :
                    rootnode = rootnode.getLeftNode()
                code = code[1:]
        # 写入的时候的最后一位的长度
        lastCode = code[-16:-8]
        lastLength = 0
        for i in range(8):
            lastLength = lastLength<<1
            if lastCode[i] =='1':
                lastLength = lastLength|1
        code = code [:-16]+code[-8:-8+lastLength]
        while len(code)>0:
            if rootnode.isleaf():
                tem_byte = six.int2byte(rootnode.getValue())
                output.write(tem_byte)
                rootnode = tem.getRoot()
            if code[0] == '1':
                rootnode = rootnode.getRightNode()
            else :
                rootnode = rootnode.getLeftNode()
            code = code[1:]
        if rootnode.isleaf():
            tem_byte = six.int2byte(rootnode.getValue())
            output.write(tem_byte)
            rootnode = tem.getRoot()
        output.close()
        self.label2.setPixmap(QPixmap('tenth_2.bmp'))   
    def ci1(self):
        global name
        f = open(name,'rb')
        filedata = f.read()
        filesize = f.tell()

        node = {}

        for i in range(filesize):
            pointnum = filedata[i]
            if pointnum in node.keys():
                node[pointnum] += 1
            else:
                node[pointnum] = 1

        hufftree = []
        
        for x in node.keys():
            tem = HuffT(0,x,node[x],None,None)
            hufftree.append(tem)
        length = len(node.keys())
        output = open('tenth_1','wb')

        a4 = length&255
        length = length>>8
        a3 = length&255
        length = length>>8
        a2 = length&255
        length = length>>8
        a1 = length&255

        output.write(six.int2byte(a1))
        output.write(six.int2byte(a2))
        output.write(six.int2byte(a3))
        output.write(six.int2byte(a4))
        for x in node.keys():
            output.write(six.int2byte(x))
            temp = node[x]
            a4 = temp&255
            temp = temp>>8
            a3 = temp&255
            temp = temp>>8
            a2 = temp&255
            temp = temp>>8
            a1 = temp&255
            output.write(six.int2byte(a1))
            output.write(six.int2byte(a2))
            output.write(six.int2byte(a3))
            output.write(six.int2byte(a4))

        finalTree = buildTree(hufftree)
        finalTree.codeTree(finalTree.getRoot(),'',node)

        code = ''

        for i in range(filesize):
            key = filedata[i]
            code = code + node[key]
            out = 0
            while (len(code)>8):
                for  i in range(8):
                    out = out<<1
                    if code [i] == '1':
                        out = out|1
                code = code[8:]
                output.write(six.int2byte(out))
                out = 0                
        # 处理剩下来的不满8位的code
        output.write(six.int2byte(len(code)))
        out = 0
        for i in range(len(code)):
            out = out<<1
            if code[i]=='1':
                out = out|1
        for i in range(8-len(code)):
            out = out<<1
        output.write(six.int2byte(out))
        output.close()
    def ct4(self):
        global name
        img = cv2.imread(name,0)
      
        h = img.shape[0]
        w = img.shape[1]

        newimg = np.zeros((h,w),np.uint8)
        for i in range(1,h-2):
            for j in range(1,w-2):
                dx = ((int(img[i-1,j-1])+int(2*img[i-1,j])+int(img[i-1,j+1]))-(int(img[i+1,j-1])+int(2*img[i+1,j])+int(img[i+1,j+1])))
                dy = ((int(img[i-1,j+1])+int(2*img[i,j+1])+int(img[i+1,j+1]))-(int(img[i-1,j-1])+int(2*img[i,j-1])+int(img[i+1,j-1])))
                newimg[i,j]=int((dx**2+dy**2)**0.5)


        length = int(np.sqrt(h**2+w**2))

        rhos=np.linspace(-length,length,int(2*length))
        thetas=np.deg2rad(np.arange(0,180))
        cos_t=np.cos(thetas)
        sin_t=np.sin(thetas)
        num_theta=len(thetas)
        vote=np.zeros((int(2*length),num_theta),dtype=np.uint64)
        y_inx,x_inx=np.nonzero(img)
        for i in range(len(x_inx)):
            x=x_inx[i]
            y=y_inx[i]
            for j in range(num_theta):
                # 整数
                rho=round(x*cos_t[j]+y*sin_t[j])+length
                if isinstance(rho,int):
                    vote[rho,j]+=1
                else:
                    vote[int(rho),j]+=1    
        t =[]
        hh ,ww= vote.shape          
        for i in range(hh):
            for j in range(ww):
                if vote[i,j]>vote.max()-10:
                    t.append([i,j])


        for i in range(len(t)):
        
            rho = rhos[t[i][0]]
            theta = thetas[t[i][1]]
            k=-np.cos(theta)/np.sin(theta)
            b=rho/np.sin(theta)

            for pt in range(w):
                j = int(pt*k+b)
                if j >h-1 or j<0:
                    pass
                else :
                    img[j,pt] = 255
        global lastname
        lastname = 'ninth_4.bmp'
        cv2.imwrite(lastname,img)
        if lastname :
            self.label2.setPixmap(QPixmap(lastname))
             

    def ct3(self):
        global name
        img = cv2.imread(name,0)

        h = img.shape[0]
        w = img.shape[1]
        newimg = np.zeros((h,w),np.uint8)
        for i in range(1,h-2):
            for j in range(1,w-2):
                laplace = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
                newimg[i,j] = abs(np.sum(img[i-1:i+2,j-1:j+2]*laplace))
        cv2.imshow("3",newimg)
        
        edges = []
        currd = 0
        counts = 0
        connect =[[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1]]

        for i in range(1,h-1):
            for j in range(1,w-1):
                edge = []
                bpt = Point(i,j)
                cpt = Point(i,j)
                if newimg[cpt.x,cpt.y]>54:
            
                    edge.clear()
                    flag = False
                    edge.append(cpt)
                    newimg[cpt.x,cpt.y] = 0
                    while flag == False:
                        counts = 0
                        while counts <8 :
                            counts+=1
                            if currd >=8 :
                                currd -=8
                            if currd <0:
                                currd +=8
                            cpt = Point(bpt.x+connect[currd][0],bpt.y+connect[currd][1])
                            if cpt.x>0 and cpt.y>0 and  cpt.x< h-1 and cpt.y<w-1:
                                if newimg[cpt.x,cpt.y]>50:
                                    currd -=2
                                    edge.append(cpt)
                                    newimg[cpt.x,cpt.y] = 0
                                    bpt.x = cpt.x
                                    bpt.y = cpt.y
                                    break
                            currd+=1
                        if 8 == counts:
                    
                            currd = 0
                            flag = True
                            
                            edges.append(edge)
                            break
        image = np.zeros((h,w),np.uint8)
        for i in range(len(edges)):
            for j in range(len(edges[i])):
                image[edges[i][j].x,edges[i][j].y]= 255
        global lastname
        lastname = 'ninth_3.bmp'
        cv2.imwrite(lastname,image)
        if lastname :
            self.label2.setPixmap(QPixmap(lastname))
    def ct2(self):
        global name
        img = cv2.imread(name,0)

        h = img.shape[0]
        w = img.shape[1]
        newimg = np.zeros((h,w),np.uint8)
        for i in range(1,h-2):
            for j in range(1,w-2):
                
                dx = ((int(img[i-1,j-1])+int(img[i-1,j])+int(img[i-1,j+1]))-(int(img[i+1,j-1])+int(img[i+1,j])+int(img[i+1,j+1])))
                dy = ((int(img[i-1,j+1])+int(img[i,j+1])+int(img[i+1,j+1]))-(int(img[i-1,j-1])+int(img[i,j-1])+int(img[i+1,j-1])))
                newimg[i,j] = int((dx**2+dy**2)**0.5)
        cv2.imshow("2",newimg)
        edges = []
        currd = 0
        counts = 0
        connect =[[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1]]

        for i in range(1,h-1):
            for j in range(1,w-1):
                edge = []
                bpt = Point(i,j)
                cpt = Point(i,j)
                if newimg[cpt.x,cpt.y]>44:
            
                    edge.clear()
                    flag = False
                    edge.append(cpt)
                    newimg[cpt.x,cpt.y] = 0
                    while flag == False:
                        counts = 0
                        while counts <8 :
                            counts+=1
                            if currd >=8 :
                                currd -=8
                            if currd <0:
                                currd +=8
                            cpt = Point(bpt.x+connect[currd][0],bpt.y+connect[currd][1])
                            if cpt.x>0 and cpt.y>0 and  cpt.x< h-1 and cpt.y<w-1:
                                if newimg[cpt.x,cpt.y]>64:
                                    currd -=2
                                    edge.append(cpt)
                                    newimg[cpt.x,cpt.y] = 0
                                    bpt.x = cpt.x
                                    bpt.y = cpt.y
                                    break
                            currd+=1
                        if 8 == counts:
                    
                            currd = 0
                            flag = True
                            
                            edges.append(edge)
                            break
        image = np.zeros((h,w),np.uint8)
        for i in range(len(edges)):
            for j in range(len(edges[i])):
                image[edges[i][j].x,edges[i][j].y]= 255
        global lastname
        lastname = 'ninth_2.bmp'
        cv2.imwrite(lastname,image)
        if lastname :
            self.label2.setPixmap(QPixmap(lastname))
    def ct1(self):
        
        global name
        img = cv2.imread(name,0)

        h = img.shape[0]
        w = img.shape[1]
        newimg = np.zeros((h,w),np.uint8)
        for i in range(1,h-2):
            for j in range(1,w-2):
                dx = ((int(img[i-1,j-1])+int(2*img[i-1,j])+int(img[i-1,j+1]))-(int(img[i+1,j-1])+int(2*img[i+1,j])+int(img[i+1,j+1])))
                dy = ((int(img[i-1,j+1])+int(2*img[i,j+1])+int(img[i+1,j+1]))-(int(img[i-1,j-1])+int(2*img[i,j-1])+int(img[i+1,j-1])))
                newimg[i,j] = int((dx**2+dy**2)**0.5)
        cv2.imshow("1",newimg)
        edges = []
        currd = 0
        counts = 0
        connect =[[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1]]

        for i in range(1,h-1):
            for j in range(1,w-1):
                edge = []
                bpt = Point(i,j)
                cpt = Point(i,j)
                if newimg[cpt.x,cpt.y]>64:
            
                    edge.clear()
                    flag = False
                    edge.append(cpt)
                    newimg[cpt.x,cpt.y] = 0
                    while flag == False:
                        counts = 0
                        while counts <8 :
                            counts+=1
                            if currd >=8 :
                                currd -=8
                            if currd <0:
                                currd +=8
                            cpt = Point(bpt.x+connect[currd][0],bpt.y+connect[currd][1])
                            if cpt.x>0 and cpt.y>0 and  cpt.x< h-1 and cpt.y<w-1:
                                if newimg[cpt.x,cpt.y]>64:
                                    currd -=2
                                    edge.append(cpt)
                                    newimg[cpt.x,cpt.y] = 0
                                    bpt.x = cpt.x
                                    bpt.y = cpt.y
                                    break
                            currd+=1
                        if 8 == counts:
                    
                            currd = 0
                            flag = True
                            
                            edges.append(edge)
                            break
        image = np.zeros((h,w),np.uint8)
        for i in range(len(edges)):
            for j in range(len(edges[i])):
                image[edges[i][j].x,edges[i][j].y]= 255
        global lastname
        lastname = 'ninth_1.bmp'
        cv2.imwrite(lastname,image)
        if lastname :
            self.label2.setPixmap(QPixmap(lastname))


    def is8(self):
        global name
        img = cv2.imread(name,0)
        plt.subplot(1,2,1),plt.imshow(img,'gray')
        img = np.double(img)
        h = img.shape[0]
        w = img.shape[1]
        rH = 2
        rL = 0.5
        c = 2
        D0=20
        hh = np.floor(h/2)
        ww = np.floor(w/2)
        img1 = np.log(img+1)
        img2 = np.fft.fft2(img1)
        
        H = np.zeros((h,w))
        for i in range(h):
            for j in range(w):
                H[i,j]=(rH-rL)*(1-np.exp(-c*((i-hh)**2+(j-ww)**2)/(D0**2)))+rL
        img4 = np.fft.ifft2(H*img2)
        img5 = np.real(np.exp(img4))
        plt.subplot(1,2,2)
        plt.imshow(img5,'gray')
        plt.show()
        
    def is7(self):
        global name
        img = cv2.imread(name,0)
        number,ok=QInputDialog.getInt(self,'','请输入模板边长')
        strline,ok2=QInputDialog.getText(self,'','请输入模板所有权值，以空格分开(数量==边长的平方)')
        num = []
        for i in strline.split():
            num.append(i)
        # print(num)    
        
        if ok and ok2 and (len(num)==number**2):
            h = img.shape[0]
            w = img.shape[1]
            newimg = np.zeros((h,w),np.uint8)
            for i in range(int(number/2),h-int(number/2)):
                for j in range(int(number/2),w-int(number/2)):
                    count = 0
                    numtemp = 0
                    for k in range(-int(number/2),int(number/2)+1):
                        for t in range(-int(number/2),int(number/2)+1):
                            numtemp+= float(img[i+k,j+t])* float(num[count])
                            count+=1
                    numtemp=int(numtemp/(number*number))
                    newimg[i,j] = numtemp
            global lastname      
            lastname = 'eighth_7.bmp'
            cv2.imwrite(lastname,newimg)
            if lastname :
                self.label2.setPixmap(QPixmap(lastname)) 
    def is6(self):
        global name
        img = cv2.imread(name,0)

        h = img.shape[0]
        w = img.shape[1]
        newimg = np.zeros((h,w),np.uint8)   
        for i in range(1,h-2):
            for j in range(1,w-2):
                laplace1 = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
                dy= np.sum(img[i-1:i+2,j-1:j+2]*laplace1)
                laplace2 = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
                dx= np.sum(img[i-1:i+2,j-1:j+2]*laplace2)
                newimg[i,j] = int((dx**2+dy**2)**0.5)
        newimg2 = newimg+img
        #cv2.imshow("6",newimg2)
        global lastname
        lastname = 'eighth_6.bmp'
        cv2.imwrite(lastname,newimg)
        if lastname :
            self.label2.setPixmap(QPixmap(lastname))
    def is5(self):
        global name
        img = cv2.imread(name,0)

        h = img.shape[0]
        w = img.shape[1]
        newimg = np.zeros((h,w),np.uint8)
        for i in range(1,h-2):
            for j in range(1,w-2):
                kernel1 = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
                kernel2 = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
                dx = np.sum(img[i-1:i+2,j-1:j+2]*kernel1)
                dy = np.sum(img[i-1:i+2,j-1:j+2]*kernel2)
                # dx = ((int(img[i-1,j-1])+int(2*img[i-1,j])+int(img[i-1,j+1]))-(int(img[i+1,j-1])+int(2*img[i+1,j])+int(img[i+1,j+1])))
                # dy = ((int(img[i-1,j+1])+int(2*img[i,j+1])+int(img[i+1,j+1]))-(int(img[i-1,j-1])+int(2*img[i,j-1])+int(img[i+1,j-1])))
                newimg[i,j] = int((dx**2+dy**2)**0.5)
        newimg2 = newimg+img
        # cv2.imshow("5",newimg2)
        global lastname
        lastname = 'eighth_5.bmp'
        cv2.imwrite(lastname,newimg)
        if lastname :
            self.label2.setPixmap(QPixmap(lastname))
    def is4(self):
        global name
        img = cv2.imread(name,0)

        h = img.shape[0]
        w = img.shape[1]
        newimg = np.zeros((h,w),np.uint8)
        for i in range(1,h-1):
            for j in range(1,w-1):
                dxy = ((int(img[i,j])-int(img[i+1,j+1]))**2+(int(img[i,j+1])-int(img[i+1,j]))**2)
                dxy = dxy**0.5
                newimg[i,j] = int(dxy)
        newimg2 = newimg+img
        cv2.imshow("4",newimg2)
        global lastname
        lastname = 'eighth_4.bmp'
        cv2.imwrite(lastname,newimg)
        if lastname :
            self.label2.setPixmap(QPixmap(lastname))
    def is3(self):
        global name
        img = cv2.imread(name,0)
        newimg = img
        number,ok=QInputDialog.getInt(self,'K邻域','输入宽度(建议>=5)')
        number2,ok2=QInputDialog.getInt(self,'K邻域','输入K')
        if ok and ok2 and (number2<(number*number-1)):
            h = img.shape[0]
            w = img.shape[1]
            for i in range(int(number/2),h-int(number/2)):
                for j in range(int(number/2),w-int(number/2)):
                    num = []
                    for k in range(-int(number/2),int(number/2)+1):
                        for t in range(-int(number/2),int(number/2)+1):
                            if k==0 and t ==0:
                                pass
                            else:
                                tempt = float(img[i+k,j+t])-float(img[i,j])
                                num.append(tempt)
           
                    num.sort()
                    numsum = 0
                    if num[0]>0:
                        for kt in range(number2):
                            numsum+=num[kt]
                    elif num[-1]<0:
                        for kt in range(number2):
                            numsum+=num[-1-kt]
                    else :
                        numt = []
                        for k in range(number*number-1):
                            numt.append(abs(num[k]))
                        numt.sort()
                        for k in range(number2):
                            for t in range(number*number-1):
                                if abs(num[t])==numt[k]:
                                    numsum +=num[t]
                                    break
                    numsum = float(numsum)
                    numsum /= number2
                    newimg[i,j] = int(img[i,j]+numsum)
            global lastname       
            lastname = 'eighth_3.bmp'
            cv2.imwrite(lastname,newimg)
            if lastname :
                self.label2.setPixmap(QPixmap(lastname))
    def is2(self):
        global name
        img = cv2.imread(name,0)
        
        number,ok=QInputDialog.getInt(self,'中值','输入宽度')
        if ok:
            h = img.shape[0]
            w = img.shape[1]
            newimg = np.zeros((h,w),np.uint8)
            for i in range(int(number/2),h-int(number/2)):
                for j in range(int(number/2),w-int(number/2)):
                    num = []
                    for k in range(-int(number/2),int(number/2)+1):
                        for t in range(-int(number/2),int(number/2)+1):
                            
                            num.append(img[i+k,j+t])
                    num.sort()
                    
                    newimg[i,j] = num[int((number*number)/2)+1]
            global lastname
            lastname = 'eighth_2.bmp'
            cv2.imwrite(lastname,newimg)
            if lastname :
                self.label2.setPixmap(QPixmap(lastname))
    def is1(self):
        global name
        img = cv2.imread(name,0)
        
        number,ok=QInputDialog.getInt(self,'均值','输入宽度')

        if ok:
            h = img.shape[0]
            w = img.shape[1]
            newimg = np.zeros((h,w),np.uint8)
            for i in range(int(number/2),h-int(number/2)):
                for j in range(int(number/2),w-int(number/2)):
                    num = 0
                    for k in range(-int(number/2),int(number/2)+1):
                        for t in range(-int(number/2),int(number/2)+1):
                            if k==0 and t==0:
                                pass
                            else:
                                num +=img[i+k,j+t]
                    num= int(num/(number*number-1))        
                    newimg[i,j] = num
            global lastname
            lastname = 'eighth_1.bmp'
            cv2.imwrite(lastname,newimg)
            if lastname :
                self.label2.setPixmap(QPixmap(lastname))    




    def cp2(self):
        global name
        img = cv2.imread(name)
        h = img.shape[0]
        w = img.shape[1]

        newimg = np.zeros((h,w,3),np.uint8)
        count = np.zeros((4096,2))
        
        for i in range(h):
            for j in range(w):
                num= 0
                for k in range(3):
                    num+=((int(img[i,j][k]/16))*(16**k))

                num = int(num)
                count[num][0]+=1
        for i in range(4096):
            count[i][1] = i
        # 先排后面再排前面，返回一维数组索引号
        idx = np.lexsort([count[:,1],-1*count[:,0]])
        sorted_count = count[idx,:]
        flag = 0
        for i in range(256,4096):
            if sorted_count[i][0]==0:
                flag = i
                break
        count_flag = np.zeros(4096)
        for i in range(256,flag):
            min = sys.maxsize
            for j in range(0,256):
                if ((sorted_count[i][1]-sorted_count[j][1])**2)**0.5<min:
                    count_flag[i] = j
                    min = ((sorted_count[i][1]-sorted_count[j][1])**2)**0.5
        for i in range(h):
            for j in range(w):
                num= 0
                for k in range(3):
                    num+=((int(img[i,j][k]/16))*(16**k))
                for t in range(4096):
                    if num == sorted_count[t][1]:
                        if t <256:
                            for k in range(3):
                                newimg[i,j][k] = (img[i,j][k]/16)*16
                        else :
                            for k in range(3):
                                newimg[i,j][k] = (int((sorted_count[int(count_flag[t])][1]/(16**k)))%16)*16
        global lastname
        lastname = "ninth_2.bmp"
        cv2.imwrite(lastname,newimg)
        if lastname :
            self.label2.setPixmap(QPixmap(lastname))          
          

    def cp1(self):
        global name
        img = cv2.imread(name)
        h = img.shape[0]
        w = img.shape[1]
        
        newimg = np.zeros((h,w,1),np.uint8)
        for i in range(h):
            for j in range(w):
                b= img[i,j,0]
                g= img[i,j,1]
                r= img[i,j,2]
                newimg[i,j] = np.uint8((r*299+g*587+b*114+500)/1000)
        global lastname
        lastname = "ninth_1.bmp"
        cv2.imwrite(lastname,newimg)
        if lastname :
            self.label2.setPixmap(QPixmap(lastname))    
    def imgt7(self):
        global name
        img = cv2.imread(name,0)
        
        coeffs = pywt.dwt2(img, 'haar')
        p1, (p2, p3, p4) = coeffs
        cv2.imwrite('seventh_7.bmp',p1)
        cv2.imwrite('seventh_8.bmp',p2)
        cv2.imwrite('seventh_9.bmp',p3)
        cv2.imwrite('seventh_10.bmp',p4)


        plt.subplot(141),plt.imshow(p1,'gray')
        plt.axis('off')
        plt.subplot(142),plt.imshow(p2,'gray')
        plt.axis('off')
        plt.subplot(143),plt.imshow(p3,'gray')
        plt.axis('off')
        plt.subplot(144),plt.imshow(p4,'gray')
        plt.axis('off')
        plt.show()

    def imgt6(self):
        global name
        img = cv2.imread(name,0)
        h = img.shape[0]
        w = img.shape[1]
        hh = int(h/2)
        ww = int(w/2)
        mask1 = np.zeros((h,w),np.uint8)
        mask1[0:80,0:80] = 1
        mask2 = np.ones((h,w),np.uint8)
        mask2[0:10,0:10] = 0
        mask = mask2*mask1
        img = np.float32(img)
        dimg = cv2.dct(img)
        dcimg = dimg*mask
        ddimg = cv2.idct(dcimg)
       
        cv2.imwrite('seventh_6.bmp',ddimg)
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        plt.subplot(121),plt.imshow(img,'gray')
        plt.axis('off')
        cv2.imwrite('seven_6_1.bmp',dimg)
        image = cv2.imread('seven_6_1.bmp')
        cv2.imshow('img',image)
        plt.subplot(122),plt.imshow(ddimg,'gray')
        plt.axis('off')
    
        plt.show()
    def imgt5(self):
        global name
        img = cv2.imread(name,0)
        h = img.shape[0]
        w = img.shape[1]
        hh = int(h/2)
        ww = int(w/2)
        
        mask1 = np.zeros((h,w),np.uint8)
        mask1[hh-80:hh+80,ww-80:ww+80] = 1
        mask2 = np.ones((h,w),np.uint8)
        mask2[hh-10:hh+10,ww-10:ww+10] = 0
        mask = mask2*mask1

        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        fimg = np.log(np.abs(fshift))
        fshift = fshift*mask
        ffshift = np.fft.ifftshift(fshift)
        ffimg = np.fft.ifft2(ffshift)
        ffimg = np.abs(ffimg)
       

        cv2.imwrite('seventh_3.bmp',ffimg)
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        plt.subplot(131),plt.imshow(img,'gray'),plt.title('原图')
        plt.axis('off')
        plt.subplot(132),plt.imshow(fimg,'gray'),plt.title('频谱图')
        plt.axis('off')
        plt.subplot(133),plt.imshow(ffimg,'gray'),plt.title('结果图')
        plt.axis('off')
    
        plt.show()
    def imgt4(self):
        global name
        img = cv2.imread(name,0)
        h = img.shape[0]
        w = img.shape[1]
        hh = int(h/2)
        ww = int(w/2)
        mask = np.zeros((h,w),np.uint8)
        mask[0:30,0:30] = 1
        img = np.float32(img)
        dimg = cv2.dct(img)
        dcimg = dimg*mask

        ddimg = cv2.idct(dcimg)
       
        cv2.imwrite('seventh_5.bmp',ddimg)
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        plt.subplot(121),plt.imshow(img,'gray')
        plt.axis('off')
        cv2.imwrite('seven_5_1.bmp',dimg)
        image = cv2.imread('seven_5_1.bmp')
        cv2.imshow('img',image)
        plt.subplot(122),plt.imshow(ddimg,'gray')
        plt.axis('off')
    
        plt.show()
    def imgt3(self):
        global name
        img = cv2.imread(name,0)
        h = img.shape[0]
        w = img.shape[1]
        hh = int(h/2)
        ww = int(w/2)
        mask = np.ones((h,w),np.uint8)
        mask[0:30,0:30] = 0
        img = np.float32(img)
        dimg = cv2.dct(img)
        dcimg = dimg*mask
        ddimg = cv2.idct(dcimg)
        cv2.imwrite('seventh_4.bmp',ddimg)
        
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        plt.subplot(121),plt.imshow(img,'gray')
        plt.axis('off')
        # plt.subplot(132),plt.imshow(dimg)
        # plt.axis('off')
        cv2.imwrite('seven_4_1.bmp',dimg)
        image = cv2.imread('seven_4_1.bmp')
        cv2.imshow('img',image)
        plt.subplot(122),plt.imshow(ddimg,'gray')
        plt.axis('off')
    
        plt.show()
    def imgt2(self):
        global name
        img = cv2.imread(name,0)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        fimg = np.log(np.abs(fshift))
       
        h = img.shape[0]
        w = img.shape[1]
        mask = np.zeros((h,w),np.uint8)
        h = int(h/2)
        w = int(w/2)

        mask[h-30:h+30,w-30:w+30] = 1
        fshift = fshift*mask
        ffshift = np.fft.ifftshift(fshift)
        ffimg = np.fft.ifft2(ffshift)
        ffimg = np.abs(ffimg)

        cv2.imwrite('seventh_2.bmp',ffimg)
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        plt.subplot(131),plt.imshow(img,'gray'),plt.title('原图')
        plt.axis('off')
        plt.subplot(132),plt.imshow(fimg,'gray'),plt.title('频谱图')
        plt.axis('off')
        plt.subplot(133),plt.imshow(ffimg,'gray'),plt.title('结果图')
        plt.axis('off')
    
        plt.show()

    def imgt1(self):
        global name
        img = cv2.imread(name,0)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        fimg = np.log(np.abs(fshift))
       
        h = img.shape[0]
        w = img.shape[1]
        mask = np.ones((h,w),np.uint8)
        h = int(h/2)
        w = int(w/2)

        mask[h-30:h+30,w-30:w+30] = 0
        fshift = fshift*mask
        ffshift = np.fft.ifftshift(fshift)
        ffimg = np.fft.ifft2(ffshift)
        ffimg = np.abs(ffimg)
        cv2.imwrite('seven_11.bmp',fimg)
        cv2.imwrite('seventh_1.bmp',ffimg)
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        plt.subplot(131),plt.imshow(img,'gray'),plt.title('原图')
        plt.axis('off')
        plt.subplot(132),plt.imshow(fimg,'gray'),plt.title('频谱图')
        plt.axis('off')
        plt.subplot(133),plt.imshow(ffimg,'gray'),plt.title('结果图')
        plt.axis('off')
    
        plt.show()

    def geotwelfth(self):
        imgName1,imgType1 = QFileDialog.getOpenFileName(self,"","")
        for index, ch in enumerate(imgName1):
            if ch =="/":
                num = index
        name1 = './'+imgName1[num+1:]
        imgName2,imgType2 = QFileDialog.getOpenFileName(self,"","")
        for index, ch in enumerate(imgName2):
            if ch =="/":
                num = index
        name2 = './'+imgName2[num+1:]
      
        img1 = cv2.imread(name1)
        h1 = img1.shape[0]
        w1 = img1.shape[1]
        img2 = cv2.imread(name2)
        h2 = img2.shape[0]
        w2 = img2.shape[1]
        hh = min(h1,h2)
        ww = min(w1,w2)
        newimg = np.zeros((hh,ww,3),np.uint8)
        for i in range(hh):
            for j in range(ww):
                newimg[i,j] = 0.7*img1[i,j]+0.3*img2[i,j]
        lastname = 'sixth_12_1.bmp'
        cv2.imwrite(lastname,newimg)
        for i in range(hh):
            for j in range(ww):
                newimg[i,j] = 0.5*img1[i,j]+0.5*img2[i,j]
        lastname = 'sixth_12_2.bmp'
        cv2.imwrite(lastname,newimg)
        for i in range(hh):
            for j in range(ww):
                newimg[i,j] = 0.3*img1[i,j]+0.7*img2[i,j]
        lastname = 'sixth_12_3.bmp'
        cv2.imwrite(lastname,newimg)
        plt.subplot(151),plt.imshow(cv2.imread(name1))
        plt.axis('off')
        plt.subplot(152),plt.imshow(cv2.imread('./sixth_12_1.bmp'))
        plt.axis('off')
        plt.subplot(153),plt.imshow(cv2.imread('./sixth_12_2.bmp'))
        plt.axis('off')
        plt.subplot(154),plt.imshow(cv2.imread('./sixth_12_3.bmp'))
        plt.axis('off')
        plt.subplot(155),plt.imshow(cv2.imread(name2))
        plt.axis('off')
        plt.show()

    def geoeleventh(self):
        global name
        number,ok=QInputDialog.getDouble(self,'右移','输入移动像素')
        if ok:
            img = cv2.imread(name)
            number1 = int(number)
            u = number-number1
            h = img.shape[0]
            w = img.shape[1]
            newimg = np.zeros((h,w,3),dtype = np.uint8)
            for i in range(h):
                for j in range(w-number1):
                    newimg[i,min(j+number1,w-1)] = (1-u)*img[i,j]+u*img[i,j+1]
            global lastname
            lastname = 'sixth_11.bmp'
            cv2.imwrite(lastname,newimg)
            if lastname :
                self.label2.setPixmap(QPixmap(lastname))    
            self.hisact(lastname)
    def geotenth(self):
        global name
        number,ok=QInputDialog.getDouble(self,'左移','输入移动像素')
        if ok:
            img = cv2.imread(name)
            number1 = int(number)
            v = number - number1 
            h = img.shape[0]
            w = img.shape[1]
                    
            # newimg[i,j] = (1-u)*(1-v)*img[hh,ww]+u*v*img[sh,sw]+(1-u)*v*img[hh,sw]+(1-v)*u*img[sh,ww]            
            newimg = np.zeros((h,w,3),dtype = np.uint8)
            for i in range(h):
                for j in range(w-number1):

                    newimg[i,j] = (1-v)*img[i,j+number1]+v*img[i,min(j+number1+1,w-1)]
            global lastname
            lastname = 'sixth_10.bmp'
            cv2.imwrite(lastname,newimg)
            if lastname :
                self.label2.setPixmap(QPixmap(lastname))    
            self.hisact(lastname)
    def geoninth(self):
        global name
        number,ok=QInputDialog.getDouble(self,'下移','输入移动像素')
        if ok:
            img = cv2.imread(name)
            number = int(number)
            h = img.shape[0]
            w = img.shape[1]
            newimg = np.zeros((h,w,3),dtype = np.uint8)
            for i in range(h-number):
                for j in range(w):

                    newimg[i+number,j] = img[i,j]
            global lastname
            lastname = 'sixth_9.bmp'
            cv2.imwrite(lastname,newimg)
            if lastname :
                self.label2.setPixmap(QPixmap(lastname))    
            self.hisact(lastname)
    def geoeigth(self):
        global name
        number,ok=QInputDialog.getDouble(self,'上移','输入移动像素')
        if ok:
            img = cv2.imread(name)
            number = int(number)
            h = img.shape[0]
            w = img.shape[1]
            newimg = np.zeros((h,w,3),dtype = np.uint8)
            for i in range(h-number):
                for j in range(w):

                    newimg[i,j] = img[i+number,j]
            global lastname
            lastname = 'sixth_8.bmp'
            cv2.imwrite(lastname,newimg)
            if lastname :
                self.label2.setPixmap(QPixmap(lastname))    
            self.hisact(lastname)
    
    def geosixth(self):
        global name
        number,ok=QInputDialog.getInt(self,'旋转','输入旋转角度(正值)')
        if ok:
            img = cv2.imread(name)
            h = img.shape[0]
            w = img.shape[1]

            angle = (float(number%360))/180*np.pi
            w_new = int(w*abs(np.cos(angle)) + h*abs(np.sin(angle)))+1
            h_new = int(w*abs(np.sin(angle)) + h*abs(np.cos(angle)))+1

            
            t = np.array([[1,0,0],[0,-1,0],[-0.5*w_new,0.5*h_new,1]])
            t = t.dot(np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]]))
            t = t.dot(np.array([[1,0,0],[0,-1,0],[0.5*w,0.5*h,1]]))
            newimg = np.zeros((h_new,w_new,3),dtype = np.uint8)
            

            for i in range(h_new):
                for j in range(w_new):
                    temp = np.array([i,j,1])
                    temp = temp.dot(t)
                    
                    hh = int(temp[0])
                    ww = int(temp[1])
                    u = temp[0] - hh
                    v = temp[1] - ww
                    sh = min(hh+1,h-1)
                    sw = min(w-1,ww+1)
                    if hh >= 0 and hh < h and ww >= 0 and ww < w:
                        newimg[i,j] = (1-u)*(1-v)*img[hh,ww]+u*v*img[sh,sw]+(1-u)*v*img[hh,sw]+(1-v)*u*img[sh,ww]
            global lastname
            lastname = 'sixth_6.bmp'
            cv2.imwrite(lastname,newimg)
            if lastname :
                self.label2.setPixmap(QPixmap(lastname))
            self.hisact(lastname)
    def geofifth(self):
        global name
        number,ok=QInputDialog.getInt(self,'旋转','输入旋转角度(正值)')
        if ok:
            img = cv2.imread(name)
            h = img.shape[0]
            w = img.shape[1]

            angle = (float(number%360))/180*np.pi
            w_new = int(w*abs(np.cos(angle)) + h*abs(np.sin(angle)))+1
            h_new = int(w*abs(np.sin(angle)) + h*abs(np.cos(angle)))+1

            
            t = np.array([[1,0,0],[0,-1,0],[-0.5*w_new,0.5*h_new,1]])
            t = t.dot(np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]]))
            t = t.dot(np.array([[1,0,0],[0,-1,0],[0.5*w,0.5*h,1]]))
            newimg = np.zeros((h_new,w_new,3),dtype = np.uint8)
            

            for i in range(h_new):
                for j in range(w_new):
                    temp = np.array([i,j,1])
                    temp = temp.dot(t)
                    u = int(temp[0])
                    v = int(temp[1])
                    if u >= 0 and u < h and v >= 0 and v < w:
                        newimg[i,j] = img[u,v]
            global lastname
            lastname = 'sixth_5.bmp'
            cv2.imwrite(lastname,newimg)
            if lastname :
                self.label2.setPixmap(QPixmap(lastname))
            self.hisact(lastname)
    def geofouth(self):
        global name
        number,ok=QInputDialog.getDouble(self,'缩小','输入缩小为')
        if ok:
            img = cv2.imread(name)
            h = img.shape[0]
            w = img.shape[1]
            h_new= int(h*number)
            w_new = int(w*number)
            newimg = np.zeros((h_new,w_new,3),dtype = np.uint8)
            for i in range(h_new):
                for j in range(w_new):
                    hh = int(i/number)
                    ww = int(j/number)
                    u = i/number - hh
                    v = j/number -ww
                    sh = min(hh+1,h-1)
                    sw = min(w-1,ww+1)
                    newimg[i,j] = (1-u)*(1-v)*img[hh,ww]+u*v*img[sh,sw]+(1-u)*v*img[hh,sw]+(1-v)*u*img[sh,ww]
            global lastname
            lastname = 'sixth_4.bmp'   
            cv2.imwrite(lastname,newimg)
            if lastname :
                self.label2.setPixmap(QPixmap(lastname))
            self.hisact(lastname)
    def geothree(self):
        global name
        number,ok=QInputDialog.getDouble(self,'放大','输入放大至')
        if ok:
            img = cv2.imread(name)
            h = img.shape[0]
            w = img.shape[1]
            h_new= int(h*number)
            w_new = int(w*number)
            newimg = np.zeros((h_new,w_new,3),dtype = np.uint8)
            for i in range(h_new):
                for j in range(w_new):
                    hh = int(i/number)
                    ww = int(j/number)
                    u = i/number - hh
                    v = j/number -ww
                    sh = min(hh+1,h-1)
                    sw = min(w-1,ww+1)
                    newimg[i,j] = (1-u)*(1-v)*img[hh,ww]+u*v*img[sh,sw]+(1-u)*v*img[hh,sw]+(1-v)*u*img[sh,ww]
            global lastname
            lastname = 'sixth_3.bmp'   
            cv2.imwrite(lastname,newimg)
            if lastname :
                self.label2.setPixmap(QPixmap(lastname))
            self.hisact(lastname)
             

    def geotwo(self):
        global name
        number,ok=QInputDialog.getDouble(self,'缩小','输入缩小为')
        if ok:
            img = cv2.imread(name)
            h = img.shape[0]
            w = img.shape[1]
            h_new = int(h*number)
            w_new = int(w*number)
            newimg = np.zeros((h_new,w_new,3),dtype = np.uint8)
            for i in range(h_new):
                for j in range(w_new):
                    hh = int(i/number)
                    ww = int(j/number)
                    newimg[i,j] = img[hh,ww]
            global lastname
            lastname = 'sixth_2.bmp'   
            cv2.imwrite(lastname,newimg)
            if lastname :
                self.label2.setPixmap(QPixmap(lastname))
            self.hisact(lastname)
    def geoone(self):
        global name
        number,ok=QInputDialog.getDouble(self,'放大','输入放大至')
        if ok :
            img = cv2.imread(name)
            h = img.shape[0]
            w = img.shape[1]
            h_new = int(h*number)
            w_new = int(w*number)
            newimg = np.zeros((h_new,w_new,3),dtype = np.uint8)
            for i in range(h_new):
                for j in range(w_new):
                    hh = int(i/number)
                    ww = int(j/number)
                    newimg[i,j] = img[hh,ww]
            global lastname
            lastname = 'sixth_1.bmp'   
            cv2.imwrite(lastname,newimg)
            if lastname :
                self.label2.setPixmap(QPixmap(lastname))
            self.hisact(lastname)     

    def equals(self):
        global name
        img = cv2.imread(name,0)
        h = img.shape[0]
        w = img.shape[1]
        image = np.zeros((h,w),np.int32)
        dist = np.zeros(256,np.float32)
        for  i in range(h):
            for j in range(w):
                pv = img[i,j]
                dist[pv]+=1
        
        for i in range(1,256):
            dist[i]=0.6*dist[i-1]+0.4*dist[i]

        for i in range(1,256):
            dist[i]+=dist[i-1]
        dist2 = np.zeros(256,np.float32)
        for i in range(0,256):
            dist2[i] = float(dist[i])/(h*w)*255
        for i  in range(h):
            for j in range(w):
                image[i][j] = dist2[img[i][j]]
        lastname1 = 'fifth_6.bmp'
        cv2.imwrite(lastname1,image) 
         
        self.hisact(lastname1)
        self.equal()
        if lastname1 :
            self.label2.setPixmap(QPixmap(lastname1))
    def equal(self):
        global name
        img = cv2.imread(name,0)
        h = img.shape[0]
        w = img.shape[1]
        image = np.zeros((h,w),np.int32)
        dist = np.zeros(256,np.int32)
        for  i in range(h):
            for j in range(w):
                pv = img[i,j]
                dist[pv]+=1
        for i in range(1,256):
            dist[i]+=dist[i-1]
        dist2 = np.zeros(256,np.float32)
        for i in range(0,256):
            dist2[i] = float(dist[i])/(h*w)*255
        for i  in range(h):
            for j in range(w):
                image[i][j] = dist2[img[i][j]]
        global lastname
        lastname = 'fifth_5.bmp'
        cv2.imwrite(lastname,image) 
        if lastname :
            self.label2.setPixmap(QPixmap(lastname))  
        self.hisact(name) 
        self.hisact(lastname)  
    def hisact2(self):
        global lastname
        if lastname :
            self.hisact(lastname)
    def hisact1(self):
        global name
        self.hisact(name)
    def point1(self):
        global name
        img = cv2.imread(name,flags=0)
        h = img.shape[0]
        w = img.shape[1]

        newimg = np.zeros((h,w,3),np.uint8)
        for i in range(h):
            for j in range(w):
                newimg[i,j] = min(1.5*img[i,j],255)
        global lastname
        lastname = 'fifth_1.bmp'
        cv2.imwrite(lastname,newimg)
        if lastname :
            self.label2.setPixmap(QPixmap(lastname))
        self.hisact(name)
        self.hisact(lastname)

    def point2(self):
        global name
        img = cv2.imread(name,flags=0)
        h = img.shape[0]
        w = img.shape[1]

        newimg = np.zeros((h,w,3),np.uint8)
        for i in range(h):
            for j in range(w):
                newimg[i,j] = 0.8*img[i,j]
        global lastname
        lastname = 'fifth_2.bmp'
        cv2.imwrite(lastname,newimg)
        if lastname :
            self.label2.setPixmap(QPixmap(lastname))
        self.hisact(name)
        self.hisact(lastname)
    def point3(self):
        global name
        img = cv2.imread(name,flags=0)
        h = img.shape[0]
        w = img.shape[1]

        newimg = np.zeros((h,w,3),np.uint8)
        for i in range(h):
            for j in range(w):
                x = img[i,j]
                newimg[i,j] = min(0.8*x*(255-x)/255+x,255)
        global lastname
        lastname = 'fifth_3.bmp'
        cv2.imwrite(lastname,newimg)
        if lastname :
            self.label2.setPixmap(QPixmap(lastname))    
        self.hisact(name)
        self.hisact(lastname)
    def point4(self):
        global name
        img = cv2.imread(name,flags=0)
        h = img.shape[0]
        w = img.shape[1]

        newimg = np.zeros((h,w,3),np.uint8)
        for i in range(h):
            for j in range(w):
                x = img[i,j]
                newimg[i,j] = min(1.2*x*(255-x)/255+x,255)
        global lastname
        lastname = 'fifth_4.bmp'
        cv2.imwrite(lastname,newimg)
        if lastname :
            self.label2.setPixmap(QPixmap(lastname))    
        self.hisact(name)
        self.hisact(lastname)
    def cut(self):
        global name
        number,ok=QInputDialog.getInt(self,'阈值化','输入阈值')
        if number >255 or number <0:
            pass
        else:
            img = cv2.imread(name,flags=0)
            h = img.shape[0]
            w = img.shape[1]

            for i in range (h):
                for j in range(w):
                    if img[i,j]>number:
                        img[i,j] = 255
                    else :
                        img[i,j] = 0
            global lastname
            lastname = 'fourth_'+str(number)+'.bmp'
            cv2.imwrite(lastname,img)  
            if lastname :
                self.label2.setPixmap(QPixmap(lastname))
            self.hisact(lastname)


    def bmp2txt(self):
        global name
        img = cv2.imread(name)

        height = img.shape[0]
        width = img.shape[1]

        with open('bmp2txt.txt','w') as f:
            for i in range(height):
                for j in range(width):
                    count = 0
                    for k in range(3):
                        count +=img[i,j][k]
                    if count >383 :
                        f.write('$')
                        print('$',end="")
                    else:
                        f.write(' ')
                        print(' ',end= "")
                f.write('\n')
                print('\n')
        f.close()


    def w(self):
        global name
        img = cv2.imread(name)
        img = np.array(img)
        for i in range (8):
            IMG=[]
            IMG = img%2
            IMG = IMG*255
            img = img/2
            img = img.astype(np.uint8)
            cv2.imwrite('third_'+str(i+1)+'.bmp',IMG)
        plt.subplot(241),plt.imshow(cv2.imread('./third_1.bmp'))
        plt.subplot(242),plt.imshow(cv2.imread('./third_2.bmp'))
        plt.subplot(243),plt.imshow(cv2.imread('./third_3.bmp'))
        plt.subplot(244),plt.imshow(cv2.imread('./third_4.bmp'))
        plt.subplot(245),plt.imshow(cv2.imread('./third_5.bmp'))
        plt.subplot(246),plt.imshow(cv2.imread('./third_6.bmp'))
        plt.subplot(247),plt.imshow(cv2.imread('./third_7.bmp'))
        plt.subplot(248),plt.imshow(cv2.imread('./third_8.bmp'))
      
        plt.show()
            

    def lh2(self):
        global name
        img = cv2.imread(name)
        num = 0
        
        height = img.shape[0]
        width = img.shape[1]

        image = np.zeros((height, width, 3), np.uint8)

   
        for i in range(height):
            for j in range(width):
                for k in range(3): 
                    if img[i, j][k] < 128:
                        color = 0
                    else :
                        color = 255
                    image[i, j][k] = np.uint8(color)   
        global lastname
        lastname =  "second_3.bmp"
        cv2.imwrite(lastname,image)  
        imgName = lastname
        if imgName :
            self.label2.setPixmap(QPixmap(imgName))
    def lh4(self):
        
        global name
        img = cv2.imread(name)

    
        height = img.shape[0]
        width = img.shape[1]

   
        image = np.zeros((height, width, 3), np.uint8)

   
        for i in range(height):
            for j in range(width):
                for k in range(3): 
                    if img[i, j][k] < 64:
                        color = 0
                    elif img[i, j][k] < 128:
                        color = 64
                    elif img[i, j][k] < 192:
                        color = 128
                    else :
                        color = 192
                    image[i, j][k] = np.uint8(color)  
        global lastname
        lastname =  "second_2.bmp"
        cv2.imwrite(lastname,image)  
        imgName = lastname
        if imgName :
            self.label2.setPixmap(QPixmap(imgName))
    def lh8(self):
        global name
        img = cv2.imread(name)
        num = 0
        

        #获取图像高度和宽度
        height = img.shape[0]
        width = img.shape[1]

        #创建一幅图像
        image = np.zeros((height, width, 3), np.uint8)

        #图像量化操作 量化等级为2
        for i in range(height):
            for j in range(width):
                for k in range(3): #对应BGR三分量
                    if img[i, j][k] < 32:
                        color = 0
                    elif img[i, j][k] < 64:
                        color = 32
                    elif img[i, j][k] < 96:
                        color = 64
                    elif img[i, j][k] < 128:
                        color = 96
                    elif img[i, j][k] < 160:
                        color = 128
                    elif img[i, j][k] < 192:
                        color = 160
                    elif img[i, j][k] < 224:
                        color = 192
                    else:
                        color = 224
                    image[i, j][k] = np.uint8(color)
        global lastname       
        lastname =  "second_1.bmp"
        cv2.imwrite(lastname,image)  
        imgName = lastname
        if imgName :
            self.label2.setPixmap(QPixmap(imgName))
        
    def cy128(self):
        temp = 128     
        global name
        img = cv2.imread(name)    

        h = img.shape[0]
        w = img.shape[1]
        height = int (h/temp)

        width = int (w/temp)
        image = np.zeros((h, w, 3), np.uint8)


        for i in range(temp):
            y = i * height
            for j in range(temp): 
                x = j * width
                b = img[y,x][0]
                g = img[y,x][1]
                r = img[y,x][2]
                for n in range(height):
                    for m in range (width):
                        image[y+n,x+m][2] = np.uint8(r)
                        image[y+n,x+m][1] = np.uint8(g)
                        image[y+n,x+m][0] = np.uint8(b)
        global lastname
        lastname =   'second_4.bmp'
        cv2.imwrite(lastname,image)  
        imgName = lastname
        if imgName :
            self.label2.setPixmap(QPixmap(imgName))

    def cy64(self):
        temp = 64
        global name
        img = cv2.imread(name)  


        h = img.shape[0]
        w = img.shape[1]
        height = int (h/temp)

        width = int (w/temp)
        image = np.zeros((h, w, 3), np.uint8)


        for i in range(temp):
            y = i * height
            for j in range(temp): 
                x = j * width
                b = img[y,x][0]
                g = img[y,x][1]
                r = img[y,x][2]
                for n in range(height):
                    for m in range (width):
                        image[y+n,x+m][2] = np.uint8(r)
                        image[y+n,x+m][1] = np.uint8(g)
                        image[y+n,x+m][0] = np.uint8(b)
        global lastname
        lastname =   'second_5.bmp'
        cv2.imwrite(lastname,image)  
        imgName = lastname
        if imgName :
            self.label2.setPixmap(QPixmap(imgName))

    def cy32(self):
        temp = 32
        global name
        img = cv2.imread(name)  


        h = img.shape[0]
        w = img.shape[1]
        height = int (h/temp)

        width = int (w/temp)
        image = np.zeros((h, w, 3), np.uint8)

        for i in range(temp):
            y = i * height
            for j in range(temp): 
                x = j * width
                b = img[y,x][0]
                g = img[y,x][1]
                r = img[y,x][2]
                for n in range(height):
                    for m in range (width):
                        image[y+n,x+m][2] = np.uint8(r)
                        image[y+n,x+m][1] = np.uint8(g)
                        image[y+n,x+m][0] = np.uint8(b)
        global lastname
        lastname =   'second_6.bmp'
        cv2.imwrite(lastname,image)  
        imgName = lastname
        if imgName :
            self.label2.setPixmap(QPixmap(imgName))


if __name__ == "__main__":
    
    app = QApplication(sys.argv)
    ex = mainwindow()
    sys.exit(app.exec_())