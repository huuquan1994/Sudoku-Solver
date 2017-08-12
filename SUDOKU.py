#File name: SUDOKU.py
#Finding unsige cell
def FindUnsignedLocation(Board, l):
    for row in range(0,9):
        for col in range(0,9):
            if (Board[row][col] == 0):
                l[0] = row
                l[1] = col
                return True
    return False


def InRow(Board, row, num):
    for i in range(0,9):
        if (Board[row][i] == num):
            return True
    return False


def InCol(Board, col, num):
    for i in range(0,9):
        if (Board[i][col] == num):
            return True
    return False


def InBox(Board, row, col, num):
    for i in range(0,3):
        for j in range(0,3):
            if (Board[i + row][j + col] == num):
                return True
    return False


def isSafe(Board, row, col, num):
    return not InCol(Board, col, num) and not InRow(Board, row, num) and not InBox(Board, row - row % 3, col - col % 3, num)

def SolveSudoku(Board):
    l=[0,0]
    if (not FindUnsignedLocation(Board, l)):
        return True
    row = l[0]
    col = l[1]
    for num in range(1,10):
        if (isSafe(Board, row, col, num)):
            Board[row][col] = num
            if (SolveSudoku(Board)):
                return True
            Board[row][col] = 0
    return False