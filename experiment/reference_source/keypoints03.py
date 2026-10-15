"""RGB-only subpixel blue eraser localization; no calibrated geometry assumed."""
import cv2
import numpy as np

MASKS = [(85,140,40,20),(95,135,70,25),(90,130,100,30)]


def blue_support(rgb, parameters):
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    low,high,sat,value=parameters
    mask=((hsv[:,:,0]>=low)&(hsv[:,:,0]<=high)&(hsv[:,:,1]>=sat)&(hsv[:,:,2]>=value)).astype(np.uint8)
    closed=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    connected=cv2.dilate(closed,np.ones((5,5),np.uint8))
    _,labels,stats,_=cv2.connectedComponentsWithStats(connected,connectivity=8)
    if len(stats)<2:return np.zeros_like(mask),hsv
    index=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    return ((labels==index)&(mask>0)).astype(np.uint8),hsv


def keypoint_features(rgb):
    assert rgb.shape==(256,640,3)
    result=[]
    for image in [rgb[:,:320],rgb[:,320:]]:
        for parameters in MASKS:
            mask,hsv=blue_support(image,parameters)
            yy,xx=np.nonzero(mask)
            if len(xx)<3:
                result.extend([0.]*16);continue
            x=xx/320.;y=yy/256.;mx=x.mean();my=y.mean()
            dx=x-mx;dy=y-my
            result.extend([mx,my,(xx.max()-xx.min()+1)/320.,(yy.max()-yy.min()+1)/256.,
                len(xx)/(320*256),np.mean(dx*dx),np.mean(dx*dy),np.mean(dy*dy),
                *np.mean(hsv[yy,xx],axis=0)/255.,*np.mean(image[yy,xx],axis=0)/255.,
                np.sqrt(np.mean(dx*dx)+np.mean(dy*dy)),1.])
    return np.asarray(result,np.float32)


def keypoint_input(points, groups, degree=1):
    # RGB derivatives are evaluated with replicated boundaries inside each recording.
    from pls03 import pls_input
    value=pls_input(points,groups,'context2',96)
    if degree==2:
        # Pairwise products of the six subpixel centroid pairs; remaining moments retained.
        centroids=points[:,[i+j for i in range(0,96,16) for j in [0,1]]]
        square=np.column_stack([centroids[:,i]*centroids[:,j] for i in range(12) for j in range(i,12)])
        value=np.concatenate([value,square],axis=1)
    return value.astype(np.float32)
