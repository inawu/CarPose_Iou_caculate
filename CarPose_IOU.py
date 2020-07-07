#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 22:27:19 2020

@author: yina
"""

from math import sqrt
import pandas as pd 
import numpy as np
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
from shapely.geometry import Polygon
from matplotlib import pyplot as plt
from shapely import geometry
import re
from sqlalchemy import Table, Column, Float, Integer, String, MetaData, ForeignKey

root_path = os.getcwd()    
raw=pd.read_csv("carpose/data_v5.csv") 
raw['iou_fix']=raw.apply(lambda _: '', axis=1)
raw['hide_point']=raw.apply(lambda _: '', axis=1)
raw['hide_x']=raw.apply(lambda _: '', axis=1)
raw['hide_y']=raw.apply(lambda _: '', axis=1)
raw['center_dis_carpose']=raw.apply(lambda _: '', axis=1)
raw['center_dis_bounding']=raw.apply(lambda _: '', axis=1)
raw['center_dis_fix']=raw.apply(lambda _: '', axis=1)
raw['point_dis_carpose']=raw.apply(lambda _: '', axis=1)
raw['point_dis_fix']=raw.apply(lambda _: '', axis=1)
raw['point_dis_bounding']=raw.apply(lambda _: '', axis=1)
raw['x_error']=raw.apply(lambda _: '', axis=1)
raw['y_error']=raw.apply(lambda _: '', axis=1)
raw['iou_pose']=raw.apply(lambda _: '', axis=1)
raw['iou_b']=raw.apply(lambda _: '', axis=1)
raw['gtp1x']=raw['gtp1x']+9
raw['gtp2x']=raw['gtp2x']+9
raw['gtp3x']=raw['gtp3x']+9
raw['gtp4x']=raw['gtp4x']+9
raw['gtp1y']=raw['gtp1y']-10
raw['gtp2y']=raw['gtp2y']-10
raw['gtp3y']=raw['gtp3y']-10
raw['gtp4y']=raw['gtp4y']-10
#raw2=raw.to_numpy()
#已知平行四边形三个点，求第四个点
#计算两点之间的距离
def CalcEuclideanDistance(point1,point2):
    vec1 = np.array(point1)
    vec2 = np.array(point2)
    distance = np.linalg.norm(vec1 - vec2)
    return distance
#计算第四个点
def CalcFourthPoint(point1,point2,point3): #pint3为A点
    D = (point1[0]+point2[0]-point3[0],point1[1]+point2[1]-point3[1])
    return D
#三点构成一个三角形，利用两点之间的距离，判断邻边AB和AC,利用向量法以及平行四边形法则，可以求得第四个点D
def JudgeBeveling(point1,point2,point3):
    dist1 = CalcEuclideanDistance(point1,point2)
    dist2 = CalcEuclideanDistance(point1,point3)
    dist3 = CalcEuclideanDistance(point2,point3)
    dist = [dist1, dist2, dist3]
    max_dist = dist.index(max(dist))
    if max_dist == 0:
        D = CalcFourthPoint(point1,point2,point3)
    elif max_dist == 1:
        D = CalcFourthPoint(point1,point3,point2)
    else:
        D = CalcFourthPoint(point2,point3,point1)
    return D

def getIou(pts1,pts2):
 polygon1 = Polygon(pts1)
 polygon2 = Polygon(pts2)
 unionArea = polygon2.buffer(0).union(polygon1.buffer(0))
 intersectionArea = polygon2.buffer(0).intersection(polygon1.buffer(0))
 if(unionArea.area ==0 or intersectionArea.area==0):
  return 0
 iou = intersectionArea.area / unionArea.area
 return iou

#print(JudgeBeveling((0,1),(1,0),(1,1)))
#print(JudgeBeveling((5,39),(500,35),(496,17)))

def DisPoint(x1,y1,x2,y2):
    dist=sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
    return dist

lenraw=raw.shape[0]

#i=21
for i in range (0,lenraw):
    #generate hide point 1;
    p1x=raw.iloc[i,5]
    p1y=raw.iloc[i,6]
    p2x=raw.iloc[i,7]
    p2y=raw.iloc[i,8]
    p3x=raw.iloc[i,9]
    p3y=raw.iloc[i,10]
    p4x=raw.iloc[i,11]
    p4y=raw.iloc[i,12]
    b1x=raw.iloc[i,15]
    b1y=raw.iloc[i,16]
    b2x=raw.iloc[i,17]
    b2y=raw.iloc[i,18]
    b3x=raw.iloc[i,19]
    b3y=raw.iloc[i,20]
    b4x=raw.iloc[i,21]
    b4y=raw.iloc[i,22] 
    
    ph1=JudgeBeveling((p4x,p4y),(p2x,p2y),(p3x,p3y))
    ph1=JudgeBeveling((p4x,p4y),(p2x,p2y),(p3x,p3y))
    ph1x=ph1[0]
    ph1y=ph1[1]
    
    ph2=JudgeBeveling((p1x,p1y),(p4x,p4y),(p3x,p3y))
    ph2=JudgeBeveling((p1x,p1y),(p4x,p4y),(p3x,p3y))
    ph2x=ph2[0]
    ph2y=ph2[1]
    #plt.scatter(x,y)
    #plt.plot(ph2x,ph2y,"or")
    
    ph3=JudgeBeveling((p1x,p1y),(p4x,p4y),(p2x,p2y))
    ph3=JudgeBeveling((p1x,p1y),(p4x,p4y),(p2x,p2y))
    ph3x=ph3[0]
    ph3y=ph3[1]
    #plt.scatter(x,y)
    #plt.plot(ph3x,ph3y,"or")
    
    
    ph4=JudgeBeveling((p1x,p1y),(p2x,p2y),(p3x,p3y))
    ph4=JudgeBeveling((p3x,p3y),(p2x,p2y),(p1x,p1y))
    ph4x=ph4[0]
    ph4y=ph4[1]
    
#polygon for bounding box
    b1=geometry.Point(b1x,b1y)
    b2=geometry.Point(b2x,b2y)
    b3=geometry.Point(b3x,b3y)
    b4=geometry.Point(b4x,b4y)
    bounding=[b1,b3,b2,b4]
    poly_b= geometry.Polygon(bounding)
    center_b=list(poly_b.centroid.coords)[0]
    center_b_x=center_b[0]
    center_b_y=center_b[1]
   
 #polygon for orginal keypoint
    x=np.array([p4x, p3x,p2x,p1x])
    y=np.array([p4y, p3y,p2y,p1y])
    #plt.scatter(x,y)
    #plt.plot(ph1x,ph1y,"or")
    p1 = geometry.Point(p1x,p1y)
    p2 = geometry.Point(p2x,p2y)
    p3 = geometry.Point(p3x,p3y)
    p4 = geometry.Point(p4x,p4y)
    edge=max( DisPoint(p1x,p1y,p2x,p2y),DisPoint(p1x,p1y,p3x,p3y),DisPoint(p1x,p1y,p4x,p4y))
    if edge== DisPoint(p1x,p1y,p2x,p2y):
         carpose = [p1, p3, p2, p4]
         test=1
    elif  edge == DisPoint(p1x,p1y,p3x,p3y):
         carpose = [p1, p2, p3, p4]
         test=2
    elif  edge == DisPoint(p1x,p1y,p4x,p4y):
         carpose = [p1, p2, p4, p3]
         test=3
          #carpose = [p4, p3, p2, p1]
    poly_k1 = geometry.Polygon(carpose)
    center_k1=list(poly_k1.centroid.coords)[0]
    center_k1_x=center_k1[0]
    center_k1_y=center_k1[1]
#polygon for fix carpose
    ##ph1 - K2
    x2=np.array([p4x, p3x,p2x,ph1x])
    y2=np.array([p4y, p3y,p2y,ph1y])
    ph1 = geometry.Point(ph1x,ph1y)
    p2 = geometry.Point(p2x,p2y)
    p3 = geometry.Point(p3x,p3y)
    p4 = geometry.Point(p4x,p4y)
    p1 = geometry.Point(p1x,p1y)
    edge=max( DisPoint(ph1x,ph1y,p2x,p2y),DisPoint(ph1x,ph1y,p3x,p3y),DisPoint(ph1x,ph1y,p4x,p4y))
    if edge== DisPoint(ph1x,ph1y,p2x,p2y):
         carpose2 = [ph1, p3, p2, p4]
         test=1
    elif  edge == DisPoint(ph1x,ph1y,p3x,p3y):
         carpose2 = [ph1, p2, p3, p4]
         test=2
    elif  edge == DisPoint(ph1x,ph1y,p4x,p4y):
         carpose2 = [ph1, p2, p4, p3]
         test=3
    #carpose2 = [p4, p3, p2, ph1]
    poly_k2 = geometry.Polygon(carpose2)
    center_k2=list(poly_k2.centroid.coords)[0]
    center_k2_x=center_k2[0]
    center_k2_y=center_k2[1]
    plt.scatter(x2,y2)
    plt.scatter(x,y)
    #plt.plot(ph4x,ph4y,"or")
    #plt.scatter(x_gt,y_gt)
    
    
    #ph2 - K4
    x4=np.array([p4x, p3x,ph2x,p1x])
    y4=np.array([p4y, p3y,ph2y,p1x])
    ph2 = geometry.Point(ph2x,ph2y)
    edge=max( DisPoint(p1x,p1y,ph2x,ph2y),DisPoint(p1x,p1y,p3x,p3y),DisPoint(p1x,p1y,p4x,p4y))
    if edge== DisPoint(p1x,p1y,ph2x,ph2y):
         carpose4 = [p1, p3, ph2, p4]
         test=1
    elif  edge == DisPoint(p1x,p1y,p3x,p3y):
         carpose4 = [p1, ph2, p3, p4]
         test=2
    elif  edge == DisPoint(p1x,p1y,p4x,p4y):
         carpose4 = [p1, ph2, p4, p3]
         test=3
    #carpose4 = [p3, p4, ph2, p1]
    poly_k4 = geometry.Polygon(carpose4)
    center_k4=list(poly_k4.centroid.coords)[0]
    center_k4_x=center_k4[0]
    center_k4_y=center_k4[1]
    
    ##ph3 - K5
    x5=np.array([p4x, ph3x,p2x,p1x])
    y5=np.array([p4y, ph3y,p2y,p1y])
    ph3 = geometry.Point(ph3x,ph3y)
    
    edge=max( DisPoint(p1x,p1y,p2x,p2y),DisPoint(p1x,p1y,ph3x,ph3y),DisPoint(p1x,p1y,p4x,p4y))
    if edge== DisPoint(p1x,p1y,p2x,p2y):
         carpose5 = [p1, ph3, p2, p4]
         test=1
    elif  edge == DisPoint(p1x,p1y,ph3x,ph3y):
         carpose5 = [p1, p2, ph3, p4]
         test=2
    elif  edge == DisPoint(p1x,p1y,p4x,p4y):
         carpose5 = [p1, p2, p4, ph3]
         test=3
        #carpose5 = [p4, ph3, p2, p1]
    poly_k5 = geometry.Polygon(carpose5)
    center_k5=list(poly_k5.centroid.coords)[0]
    center_k5_x=center_k5[0]
    center_k5_y=center_k5[1]
    
    ##ph4 - K3
    x3=np.array([ph4x, p3x,p2x,p1x])
    y3=np.array([ph4y, p3y,p2y,p1y])
    ph4 = geometry.Point(ph4x,ph4y)
    plt.scatter(x3,y3)
    #carpose3 = [ph4, p2, p1, p3]
    edge=max( DisPoint(p1x,p1y,p2x,p2y),DisPoint(p1x,p1y,p3x,p3y),DisPoint(p1x,p1y,ph4x,ph4y))
    if edge== DisPoint(p1x,p1y,p2x,p2y):
         carpose3 = [p1, p3, p2, ph4]
         test=1
    elif  edge == DisPoint(p1x,p1y,p3x,p3y):
         carpose3 = [p1, p2, p3, ph4]
         test=2
    elif  edge == DisPoint(p1x,p1y,ph4x,ph4y):
         carpose3 = [p1, p2, ph4, p3]
         test=3
    #carpose3 = [p1, p3, ph4, p2]
    poly_k3 = geometry.Polygon(carpose3)
    center_k3=list(poly_k3.centroid.coords)[0]
    center_k3_x=center_k3[0]
    center_k3_y=center_k3[1]

 #polygon for gt
    gtp1x=raw.iloc[i,23]
    gtp1y=raw.iloc[i,24]
    gtp2x=raw.iloc[i,25]
    gtp2y=raw.iloc[i,26]
    gtp3x=raw.iloc[i,27]
    gtp3y=raw.iloc[i,28]
    gtp4x=raw.iloc[i,29]
    gtp4y=raw.iloc[i,30]
    gt1=geometry.Point(gtp1x,gtp1y)
    gt2=geometry.Point(gtp2x,gtp2y)
    gt3=geometry.Point(gtp3x,gtp3y)
    gt4=geometry.Point(gtp4x,gtp4y)
    gt=[gt4,gt1,gt2,gt3]
    poly_gt = geometry.Polygon(gt)
    center_gt=list(poly_gt.centroid.coords)[0]
    center_gt_x=center_gt[0]
    center_gt_y=center_gt[1]
    x_gt=np.array([ gtp4x, gtp1x, gtp2x, gtp3x])
    y_gt=np.array([ gtp4y, gtp1y, gtp2y, gtp3y])
    plt.scatter(x,y)
    #plt.plot(ph4x,ph4y,"or")
    plt.scatter(x_gt,y_gt)
    dist1_pose=min(DisPoint(gtp1x,gtp1y,p1x,p1y),DisPoint(gtp1x,gtp1y,p2x,p2y),DisPoint(gtp1x,gtp1y,p3x,p3y),DisPoint(gtp1x,gtp1y,p4x,p4y))
    dist2_pose=min(DisPoint(gtp2x,gtp2y,p1x,p1y),DisPoint(gtp2x,gtp2y,p2x,p2y),DisPoint(gtp2x,gtp1y,p3x,p3y),DisPoint(gtp2x,gtp2y,p4x,p4y))
    dist3_pose=min(DisPoint(gtp3x,gtp3y,p1x,p1y),DisPoint(gtp3x,gtp3y,p2x,p2y),DisPoint(gtp3x,gtp3y,p3x,p3y),DisPoint(gtp3x,gtp3y,p4x,p4y))
    dist4_pose=min(DisPoint(gtp4x,gtp4y,p1x,p1y),DisPoint(gtp4x,gtp4y,p2x,p2y),DisPoint(gtp4x,gtp4y,p3x,p3y),DisPoint(gtp4x,gtp4y,p4x,p4y))
    raw['point_dis_carpose'][i]=(dist1_pose+dist2_pose+dist3_pose+dist4_pose)/4
    dist1_b=min(DisPoint(gtp1x,gtp1y,b1x,b1y),DisPoint(gtp1x,gtp1y,b2x,b2y),DisPoint(gtp1x,gtp1y,b3x,b3y),DisPoint(gtp1x,gtp1y,b4x,b4y))
    dist2_b=min(DisPoint(gtp2x,gtp2y,b1x,b1y),DisPoint(gtp2x,gtp2y,b2x,b2y),DisPoint(gtp2x,gtp1y,b3x,b3y),DisPoint(gtp2x,gtp2y,b4x,b4y))
    dist3_b=min(DisPoint(gtp3x,gtp3y,b1x,b1y),DisPoint(gtp3x,gtp3y,b2x,b2y),DisPoint(gtp3x,gtp3y,b3x,b3y),DisPoint(gtp3x,gtp3y,b4x,b4y))
    dist4_b=min(DisPoint(gtp4x,gtp4y,b1x,b1y),DisPoint(gtp4x,gtp4y,b2x,b2y),DisPoint(gtp4x,gtp4y,b3x,b3y),DisPoint(gtp4x,gtp4y,b4x,b4y))
    raw['point_dis_bounding'][i]=(dist1_b+dist2_b+dist3_b+dist4_b)/4
    #test=(dist1_b+dist2_b+dist3_b+dist4_b)/4
    raw['center_dis_bounding'][i]=DisPoint(center_b_x,center_b_y,center_gt_x,center_gt_y)
    
    plt.fill(x_gt,y_gt,facecolor="none",edgecolor='red')
    plt.fill(x, y,facecolor="none",edgecolor='green')
     
    plt.fill(x_gt,y_gt,facecolor="none",edgecolor='red')
    plt.fill(x3, y3,facecolor="none",edgecolor='blue')
     
    iou_1=getIou(poly_k1,poly_gt)

    iou_2=getIou(poly_k2,poly_gt)
    iou_3=getIou(poly_k3,poly_gt)
    iou_4=getIou(poly_k4,poly_gt)
    iou_5=getIou(poly_k5,poly_gt)
    iou_fix=max(iou_2,iou_3,iou_4,iou_5)
    raw['iou_b'][i]=getIou(poly_b,poly_gt)
    raw['iou_pose'][i]=iou_1
    raw['iou_fix'][i]=iou_fix
    if iou_fix==iou_2:
        raw['hide_point'][i]=1
        raw['hide_x'][i]=ph1x
        raw['hide_y'][i]=ph1y
        #plt.fill(x2, y2)
        #plt.fill(x_gt,y_gt,"or")
        raw['center_dis_fix'][i]=DisPoint(center_k2_x,center_k2_y,center_gt_x,center_gt_y)
        raw['center_dis_carpose'][i]=DisPoint(center_k1_x,center_k1_y,center_gt_x,center_gt_y)
        dist1_fix=min(DisPoint(gtp1x,gtp1y,ph1x,ph1y),DisPoint(gtp1x,gtp1y,p2x,p2y),DisPoint(gtp1x,gtp1y,p3x,p3y),DisPoint(gtp1x,gtp1y,p4x,p4y))
        dist2_fix=min(DisPoint(gtp2x,gtp2y,ph1x,ph1y),DisPoint(gtp2x,gtp2y,p2x,p2y),DisPoint(gtp2x,gtp1y,p3x,p3y),DisPoint(gtp2x,gtp2y,p4x,p4y))
        dist3_fix=min(DisPoint(gtp3x,gtp3y,ph1x,ph1y),DisPoint(gtp3x,gtp3y,p2x,p2y),DisPoint(gtp3x,gtp3y,p3x,p3y),DisPoint(gtp3x,gtp3y,p4x,p4y))
        dist4_fix=min(DisPoint(gtp4x,gtp4y,ph1x,ph1y),DisPoint(gtp4x,gtp4y,p2x,p2y),DisPoint(gtp4x,gtp4y,p3x,p3y),DisPoint(gtp4x,gtp4y,p4x,p4y))
        raw['x_error'][i]=center_gt_x-center_k2_x
        raw['y_error'][i]=center_gt_y-center_k2_y
        raw['point_dis_fix'][i]=(dist1_fix+dist2_fix+dist3_fix+dist4_fix)/4
    elif iou_fix==iou_3:
        raw['hide_point'][i]=4
        raw['hide_x'][i]=ph1x
        raw['hide_y'][i]=ph1y
        #plt.fill(x3, y3)
        #plt.fill(x_gt,y_gt,"or")
        raw['center_dis_fix'][i]=DisPoint(center_k3_x,center_k3_y,center_gt_x,center_gt_y)
        raw['center_dis_carpose'][i]=DisPoint(center_k1_x,center_k1_y,center_gt_x,center_gt_y)
        dist1_fix=min(DisPoint(gtp1x,gtp1y,p1x,p1y),DisPoint(gtp1x,gtp1y,p2x,p2y),DisPoint(gtp1x,gtp1y,p3x,p3y),DisPoint(gtp1x,gtp1y,ph4x,ph4y))
        dist2_fix=min(DisPoint(gtp2x,gtp2y,p1x,p1y),DisPoint(gtp2x,gtp2y,p2x,p2y),DisPoint(gtp2x,gtp1y,p3x,p3y),DisPoint(gtp2x,gtp2y,ph4x,ph4y))
        dist3_fix=min(DisPoint(gtp3x,gtp3y,p1x,p1y),DisPoint(gtp3x,gtp3y,p2x,p2y),DisPoint(gtp3x,gtp3y,p3x,p3y),DisPoint(gtp3x,gtp3y,ph4x,ph4y))
        dist4_fix=min(DisPoint(gtp4x,gtp4y,p1x,p1y),DisPoint(gtp4x,gtp4y,p2x,p2y),DisPoint(gtp4x,gtp4y,p3x,p3y),DisPoint(gtp4x,gtp4y,ph4x,ph4y))
        raw['point_dis_fix'][i]=(dist1_fix+dist2_fix+dist3_fix+dist4_fix)/4
        raw['x_error'][i]=center_gt_x-center_k3_x
        raw['y_error'][i]=center_gt_y-center_k3_y
    elif iou_fix==iou_4:
        raw['hide_point'][i]=2
        raw['hide_x'][i]=ph1x
        raw['hide_y'][i]=ph1y
        #plt.fill(x2, y2)
        #plt.fill(x_gt,y_gt,"or")
        raw['center_dis_fix'][i]=DisPoint(center_k4_x,center_k4_y,center_gt_x,center_gt_y)
        raw['center_dis_carpose'][i]=DisPoint(center_k4_x,center_k4_y,center_gt_x,center_gt_y)
        dist1_fix=min(DisPoint(gtp1x,gtp1y,p1x,p1y),DisPoint(gtp1x,gtp1y,ph2x,ph2y),DisPoint(gtp1x,gtp1y,p3x,p3y),DisPoint(gtp1x,gtp1y,p4x,p4y))
        dist2_fix=min(DisPoint(gtp2x,gtp2y,p1x,p1y),DisPoint(gtp2x,gtp2y,ph2x,ph2y),DisPoint(gtp2x,gtp1y,p3x,p3y),DisPoint(gtp2x,gtp2y,p4x,p4y))
        dist3_fix=min(DisPoint(gtp3x,gtp3y,p1x,p1y),DisPoint(gtp3x,gtp3y,ph2x,ph2y),DisPoint(gtp3x,gtp3y,p3x,p3y),DisPoint(gtp3x,gtp3y,p4x,p4y))
        dist4_fix=min(DisPoint(gtp4x,gtp4y,p1x,p1y),DisPoint(gtp4x,gtp4y,ph2x,ph2y),DisPoint(gtp4x,gtp4y,p3x,p3y),DisPoint(gtp4x,gtp4y,p4x,p4y))
        raw['point_dis_fix'][i]=(dist1_fix+dist2_fix+dist3_fix+dist4_fix)/4
        raw['x_error'][i]=center_gt_x-center_k4_x
        raw['y_error'][i]=center_gt_y-center_k4_y
    elif iou_fix==iou_5:
        raw['hide_point'][i]=3
        raw['hide_x'][i]=ph1x
        raw['hide_y'][i]=ph1y
        plt.fill(x3, y3)
        #plt.fill(x_gt,y_gt,"or")
        #plt.show()
        raw['center_dis_fix'][i]=DisPoint(center_k5_x,center_k5_y,center_gt_x,center_gt_y)
        raw['center_dis_carpose'][i]=DisPoint(center_k5_x,center_k5_y,center_gt_x,center_gt_y)
        dist1_fix=min(DisPoint(gtp1x,gtp1y,p1x,p1y),DisPoint(gtp1x,gtp1y,p2x,p2y),DisPoint(gtp1x,gtp1y,p3x,ph3y),DisPoint(gtp1x,gtp1y,p4x,p4y))
        dist2_fix=min(DisPoint(gtp2x,gtp2y,p1x,p1y),DisPoint(gtp2x,gtp2y,p2x,p2y),DisPoint(gtp2x,gtp1y,p3x,ph3y),DisPoint(gtp2x,gtp2y,p4x,p4y))
        dist3_fix=min(DisPoint(gtp3x,gtp3y,p1x,p1y),DisPoint(gtp3x,gtp3y,p2x,p2y),DisPoint(gtp3x,gtp3y,p3x,ph3y),DisPoint(gtp3x,gtp3y,p4x,p4y))
        dist4_fix=min(DisPoint(gtp4x,gtp4y,p1x,p1y),DisPoint(gtp4x,gtp4y,p2x,p2y),DisPoint(gtp4x,gtp4y,p3x,ph3y),DisPoint(gtp4x,gtp4y,p4x,p4y))
        raw['point_dis_fix'][i]=(dist1_fix+dist2_fix+dist3_fix+dist4_fix)/4
        raw['x_error'][i]=center_gt_x-center_k5_x
        raw['y_error'][i]=center_gt_y-center_k5_y

print(raw['iou_fix'].mean (axis=0))
print(raw['iou_pose'].mean(axis=0))
print(raw['iou_b'].mean(axis=0))