import cv2
import numpy as np

print('Readig...')
## Read
img = cv2.imread("photo.jpg")

print('slicing colors...')
## convert to hsv
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


row, column, channels = img.shape

new_val = 210

beps = 35
hsv = cv2.GaussianBlur(hsv, (beps, beps), 0)


# bluring image
print(hsv.shape[0], 'x', hsv.shape[1])

used = []
for i in range(0, hsv.shape[0]):
    used.append([])
    for j in range(0, hsv.shape[1]):
        used[i].append(False)


def bfs(y, x, val, eps):
    global hsv, used
    q = [(y, x)]
    while len(q) > 0:
        x = q[0][1]
        y = q[0][0]
        used[y][x] = True
        
        hsv[y][x][0] = val
        if x + 1 < hsv.shape[1] and used[y][x + 1] == False and abs(float(hsv[y][x + 1][0]) - val) <= eps:
            q.append((y, x + 1))
            used[y][x + 1] = True
        if x - 1 >= 0 and used[y][x - 1] == False and abs(float(hsv[y][x - 1][0]) - val) <= eps:
            q.append((y, x - 1))
            used[y][x - 1] = True

        if y + 1 < hsv.shape[0] and used[y + 1][x] == False and abs(float(hsv[y + 1][x][0]) - val) <= eps:
            q.append((y + 1, x))
            used[y + 1][x] = True
        if y - 1 >= 0 and used[y - 1][x] == False and abs(float(hsv[y - 1][x][0]) - val) <= eps:
            q.append((y - 1, x))
            used[y - 1][x] = True
        del q[0]


print('\nworking with hsv...')

for i in range(0, hsv.shape[0]):
    for j in range(0, hsv.shape[1]):
        for g in range(1, 11):
            if i == hsv.shape[0] // 100 * g * 10 and j == 0:
                print(g * 10, f'%')
        if used[i][j] == False:
            bfs(i, j, hsv[i][j][0], 2.25)


mask = cv2.inRange(hsv, (0, 75, 0), (33, 255,255))

## slice the red
imask = mask>0
red = np.zeros_like(img, np.uint8)
red[imask] = img[imask]


## slice the yellow
mask = cv2.inRange(hsv, (30.00001, 75, 0), (35, 255,255))

imask = mask>0
yel = np.zeros_like(img, np.uint8)
yel[imask] = img[imask]


print('loading...')

for i in range(0, red.shape[0]):
    for j in range(0, red.shape[1]):
        for g in range(1, 11):
            if i == hsv.shape[0] // 100 * g * 10 and j == 0:
                print(g * 10, f'%')


        if (i - 1 >= 0) and (yel[i][j].all() != 0 or red[i][j].all() != 0) and (yel[i - 1][j].all() == 0 and red[i - 1][j].all() == 0):
            img.itemset((i, j, 0), 255)
            img.itemset((i, j, 1), 255)
            img.itemset((i, j, 2), 255)
        if (i + 1 < red.shape[0]) and (yel[i][j].all() != 0 or red[i][j].all() != 0) and (yel[i - 1][j].all() == 0 and red[i - 1][j].all() == 0):
            img.itemset((i, j, 0), 255)
            img.itemset((i, j, 1), 255)
            img.itemset((i, j, 2), 255)

        if (j - 1 >= 0) and (yel[i][j].all() != 0 or red[i][j].all() != 0) and (yel[i][j - 1].all() == 0 and red[i][j - 1].all() == 0):
            img.itemset((i, j, 0), 255)
            img.itemset((i, j, 1), 255)
            img.itemset((i, j, 2), 255)
        if (j + 1 < red.shape[1]) and (yel[i][j].all() != 0 or red[i][j].all() != 0) and (yel[i][j + 1].all() == 0 and red[i][j + 1].all() == 0):
            img.itemset((i, j, 0), 255)
            img.itemset((i, j, 1), 255)
            img.itemset((i, j, 2), 255)

        if yel[i][j].all() != 0:
            img.itemset((i, j, 1), new_val)
        if red[i][j].all() != 0:
            img.itemset((i, j, 2), new_val)


HSVTOBGR = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

print(hsv.shape[0], 'x', hsv.shape[1])

## save
cv2.imwrite("res.jpg", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
