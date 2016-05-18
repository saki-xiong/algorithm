#-*- coding: UTF-8 -*- 
import numpy
from numpy import*
def loaddata():
	x=[0.,1.,2.,3.,4.,5.,6.,7.,8.,9.]
	xmat=mat(x)
	classLabels=[1.,1.,1.,-1.,-1.,-1.,1.,1.,1.,-1.]
	labelmat=mat(classLabels)
	return xmat,classLabels

def treeClassify(xmat,threshVal,inequal):
	retArray=mat(ones(shape(xmat),float))
	if inequal == 'lt':
		retArray[xmat<=threshVal]=-1.0
	else:
		retArray[xmat> threshVal]=-1.0
	return retArray
	
def buildbesttree(xmat,classLabels,D):
	#xmat=mat(x)
	labelmat=mat(classLabels).T
	#print xmat,labelmat
	m=shape(xmat)[1]
	bestClassEst= mat(zeros((m,1)),float)
	rangeMin=xmat.min()
	rangeMax=xmat.max()
	stepNum=9.0
	beststump={}
	stepSize=(rangeMax-rangeMin)/stepNum
	minError=inf
	for i in range(-1,int(stepNum)+1):
		for inequal in['lt','gt']:
			threshVal=(rangeMin+0.5+float(i)*stepSize)
			predictedVal=treeClassify(xmat,threshVal,inequal)
			predictedVal=predictedVal.T
			errArr=mat(ones((m,1)))
			errArr[predictedVal == labelmat] = 0
			weightedError = D.T* errArr
			#print weightedError	
			if weightedError < minError:
				minError = weightedError
				bestClassEst = predictedVal.copy()
				beststump['ineq']=inequal
				beststump['thresh']=threshVal
			#print threshVal,"weightedError:",weightedError,bestClassEst.T,"minError:",minError
	return beststump, minError,bestClassEst

def adaboostTrain(xmat,classLabels,numt=100):
	m=shape(xmat)[1]
	aggClassEst = mat(zeros((m,1)))
	D=mat(ones((m,1))/m)
	weakClassArr = []
	for i in range(1,numt):
		bestStump, Error,classEst=buildbesttree(xmat,classLabels,D)
		#print "D:",D
		#alpha = 0.5*(log((1-Error)/Error))
		alpha = float(0.5 * log((1.0 - Error)/max(Error, 1e-16)))
		#print alpha
		bestStump['alpha'] = alpha
		weakClassArr.append(bestStump)
		#Z = multiply((mat((-1.0)* mat(alpha)) * mat(classLabels)).T, classEst)
		Z = multiply(-1 * alpha * mat(classLabels).T, classEst)
		#print Z
		D = multiply(D, exp(Z))
		D = D/D.sum()
		aggClassEst += alpha *mat(classEst)    
		aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m,1),float))
		errorRate = aggErrors.sum()/m
		#print "total error: ", errorRate, "\n"
		#print classEst.T,D.T,"alpha:",alpha,aggClassEst.T
		if errorRate == 0.0: 
			break
	print aggClassEst.T,"total error: ", errorRate	
	return weakClassArr
	
def adaClassify(dataToClass,classifierArr):
	dataMatrix=mat(dataToClass)
	m=shape(dataMatrix)[0]
	aggClassEst = mat(zeros((m,1)))
	for i in range(len(classifierArr)):
		classEst=treeClassify(dataMatrix,classifierArr[i]['thresh'],classifierArr[i]['ineq'])
		aggClassEst+=classifierArr[i]['alpha']*classEst
		print aggClassEst
	return sign(aggClassEst)