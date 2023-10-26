
import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt
from qbstyles import mpl_style

X = np.random.randint(10,85,(50,2))
Z = np.vstack((X))

Z = np.float32(Z)
# define criteria and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS , 100, 1.0)
ret,label,center=cv2.kmeans(Z,3,None,criteria,100,cv2.KMEANS_RANDOM_CENTERS)
#print(label)
#print(center)
# Now separate the data, Note the flatten()
A = Z[label.ravel()==0]
B = Z[label.ravel()==1]
C = Z[label.ravel()==2]

# Plot the data
mpl_style(dark=True)
plt.scatter(A[:,0],A[:,1],s=50,c = 'r')
plt.scatter(B[:,0],B[:,1],s=50,c = 'g')
plt.scatter(C[:,0],C[:,1],s=50,c = 'b')
plt.scatter(center[:,0],center[:,1],s = 60,c = 'y', marker = 's')
plt.xlabel('X-axis Label'),plt.ylabel('Y-axis Label')
plt.show()


