import cv2
import numpy as np
import joblib
import SUDOKU

font = cv2.FONT_HERSHEY_SIMPLEX
ratio2 = 3
kernel_size = 3
lowThreshold = 30
count = 1
# Load the pre-trained SVM model.
# Please note that you will need to train a new classifier if you run this code under Python 3.x
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
    lines = cv2.HoughLines(edges, 2, np.pi /180, 300, 0, 0)
    
    if (lines is not None):
        lines = sorted(lines, key=lambda line:line[0][0])
        diff_hor = 0
        diff_ver = 0
        lines_new=[]
        Points=[]
        for line in lines:
            rho = line[0][0]
            theta = line[0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            if (b>0.5):
                if(rho-diff_hor>10):
                    diff_hor=rho
                    # cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
                    lines_new.append([rho,theta, 0])
            else:
                if(rho-diff_ver>10):
                    diff_ver=rho
                    # cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
                    lines_new.append([rho,theta, 1])
        for i in range(len(lines_new)):
            if(lines_new[i][2] == 0):
                for j in range(len(lines_new)):
                    if (lines_new[j][2]==1):
                        theta1=lines_new[i][1]
                        theta2=lines_new[j][1]
                        p1=lines_new[i][0]
                        p2=lines_new[j][0]
                        xy = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
                        p = np.array([p1,p2])
                        res = np.linalg.solve(xy, p)
                        Points.append(res)
                        # Draw intersection points
                        # cv2.circle(frame, (res[0], res[1]), 3, (0,255,0), -1)

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
            if(SUDOKU.SolveSudoku(board)):
                # If it can solve SUDOKU matrix, show the result
                for i in range(0,9):
                    for j in range(0,9):
                        if(result[i][j]==0):
                            cv2.putText(frame,str(board[i][j]),(int(Points[j+i*10+10][0]+15), 
                                                                 int(Points[j+i*10+10][1]-10)),font,1,(225,0,0),2)
                # Show the result picture
                cv2.imshow("Result", frame)
                
    cv2.imshow("SUDOKU Solver", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # Press ESC key to exit
        break
vc.release()
cv2.destroyAllWindows()