import cv2
import numpy as np
import joblib
import SUDOKU

font = cv2.FONT_HERSHEY_SIMPLEX
ratio2 = 3
kernel_size = 3
lowThreshold = 30
count = 1
clf = joblib.load('classifier.pkl')

is_print = True

cv2.namedWindow("SUDOKU Solver")
vc = cv2.VideoCapture(0)
if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False
while rval:
    sudoku1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sudoku1 = cv2.blur(sudoku1, (1,1))
    edges = cv2.Canny(sudoku1, lowThreshold, lowThreshold*ratio2, kernel_size)
    lines = cv2.HoughLines(edges, 2, cv2.cv.CV_PI /180, 300, 0, 0)
    if (lines is not None):
        lines = lines[0]
        lines = sorted(lines, key=lambda line:line[0])
        diff_ngang = 0
        diff_doc = 0
        lines_1=[]
        Points=[]
        for rho,theta in lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            if (b>0.5):
                if(rho-diff_ngang>10):
                    diff_ngang=rho
                    lines_1.append([rho,theta, 0])
            else:
                if(rho-diff_doc>10):
                    diff_doc=rho
                    lines_1.append([rho,theta, 1])
        for i in range(len(lines_1)):
            if(lines_1[i][2] == 0):
                for j in range(len(lines_1)):
                    if (lines_1[j][2]==1):
                        theta1=lines_1[i][1]
                        theta2=lines_1[j][1]
                        p1=lines_1[i][0]
                        p2=lines_1[j][0]
                        xy = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
                        p = np.array([p1,p2])
                        res = np.linalg.solve(xy, p)
                        Points.append(res)

        if(len(Points)==100):
            result = []
            board = []
            sudoku1 = cv2.adaptiveThreshold(sudoku1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 101, 1)
            for i in range(0,9):
                for j in range(0,9):
                    y1=int(Points[j+i*10][1]+5)
                    y2=int(Points[j+i*10+11][1]-5)
                    x1=int(Points[j+i*10][0]+5)
                    x2=int(Points[j+i*10+11][0]-5)
                    X = sudoku1[y1:y2,x1:x2]
                    if(X.size!=0):
                        X = cv2.resize(X, (36,36))
                        num = clf.predict(np.reshape(X, (1,-1)))
                        
                        #Collect the result
                        result.append(num)
                        board.append(num)

            # Reshape to 9x9 matrix
            result = np.reshape(result, (9,9))
            board = np.reshape(board, (9,9))
            
            # Solve the SUDOKU grid
            if(SUDOKU.SolveSudoku(result)):
                # If it can solve SUDOKU matrix, show the result
                for i in range(0,9):
                    for j in range(0,9):
                        if(array[i][j]==0):
                            cv2.putText(frame,str(result[i][j]),(int(Points[j+i*10+10][0]+15), 
                                                                 int(Points[j+i*10+10][1]-10)),font,1,(225,0,0),2)
                # Show the result picture
                cv2.imshow("Result", frame)
                
    cv2.imshow("SUDOKU Solver", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
vc.release()
cv2.destroyAllWindows()