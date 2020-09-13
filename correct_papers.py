import cv2 as cv
import numpy as np


class imgData:
    """存放使用cv2 處理圖片的各種資料"""
    def __init__(self):
        self.img = np.array([])
        self.imgGray = np.array([])
        self.imgBlur = np.array([])
        self.imgCanny = np.array([])
        self.imgDilate = np.array([])
        self.imgErode = np.array([])
        self.contours = np.array([])  # 框框資料
        self.conArea = []  # 存放框框的面積由大到小,外接矩形的approx點 以及 cnt的點  三項資料
        self.imgContour = np.array([])
        self.imgApprox = np.array([])
        self.imgWarp = np.array([])
        self.transformPts = []  # 儲存使用cv.warpPerspective()函式時 的轉換資料
        self.transformWandH = []  # 儲存使用cv.warpPerspective()函式時 的轉換資料


def find_contours(im: imgData):
    """找框框函式"""
    im.imgGray = cv.cvtColor(im.img, cv.COLOR_BGR2GRAY)
    im.imgBlur = cv.GaussianBlur(im.imgGray, (5, 5), 1)
    im.imgCanny = cv.Canny(im.imgBlur, 100, 100)
    im.imgDilate = cv.dilate(im.imgCanny, (5, 5), 3)
    im.imgErode = cv.erode(im.imgDilate, (5, 5), 2)
    im.contours, im.hierarchy = cv.findContours(im.imgCanny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)


def get_answer_contours(contours, want_Contours_Num=-1):
    """答案欄25個選項圓框框處理"""
    # 若找到1個以上 則按照面積大小 以及want_Contours_Num 的數量 決定顯示最大的幾個邊框 預設顯示所有邊框
    if len(contours) != 1:
        newContours = []
        for cnt in contours:
            area = cv.contourArea(cnt)
            newContours.append([area, cnt])
        newContours = sorted(newContours, key=lambda x: x[0], reverse=True)  # 按面積大小排列
        newContours = list(map(lambda x: x[1], newContours))  # area不要 只取cnt部分
        newContours = newContours[:want_Contours_Num]  # 決定數量
        return newContours
    else:  # 若只找到一個邊框 則回傳所有邊框
        return contours


def reorder_myAnswerCon(myAnswerCon):
    """25個原形框框找出來後並非照順序排列 使用此函式按照row column重新排列框框"""
    newCon = sorted(myAnswerCon, key=lambda x: x[0][0][1])
    newCon = np.array(newCon)
    newCon = list(np.reshape(newCon, (5, 5)))
    newCon = list(map(lambda x: sorted(x, key=lambda y: y[0][0][0]), newCon))
    newCon = np.reshape(newCon, (25, 1))
    newCon = list(newCon)
    newCon = list(map(lambda x: x[0], newCon))
    return newCon

def find_approx(im: imgData):
    """找出矩形邊框的四個頂點"""
    im.conArea = []
    for cnt in im.contours:
        area = cv.contourArea(cnt)
        if area > 0:
            pere = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.04 * pere, True)
            if len(approx) == 4:
                approx = reorder(approx)
                im.conArea.append([area, approx, cnt])
    im.conArea = sorted(im.conArea, key=lambda x: x[0], reverse=True)


def reorder(approx):
    """重新排列四個頂點順序"""
    approx = np.resize(approx, (4, 2))
    new = np.zeros_like(approx)
    add = approx.sum(1)
    diff = np.diff(approx, axis=1)
    new[0] = approx[np.argmin(add)]
    new[3] = approx[np.argmax(add)]
    new[1] = approx[np.argmin(diff)]
    new[2] = approx[np.argmax(diff)]
    return new


def get_warp(im: imgData, select_warp_number=0):
    """取得焦點畫面"""
    find_contours(im)
    find_approx(im)
    w, h = 210 * 3, 210 * 3
    try:
        selectApprox = im.conArea[select_warp_number]
    except:
        return
    pts1 = np.float32(selectApprox[1])
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    im.transformPts = [pts1, pts2]
    im.transformWandH = [w, h]
    M = cv.getPerspectiveTransform(pts1, pts2)
    im.imgWarp = cv.warpPerspective(im.img, M, (w, h))


def inverse_warp(im: imgData, image, wANDh=False):
    """反get_warp函式"""
    pts1, pts2 = im.transformPts
    if wANDh:
        w, h = im.transformWandH
    else:
        w, h = 640, 480
    M = cv.getPerspectiveTransform(pts2, pts1)
    image = cv.warpPerspective(image, M, (w, h))
    return image


def check_Answer(newCon,image):
    count = 0
    for cnt in newCon:
        if count in myAnswer:  # 首先找出每個答案圈圈的中心點
            leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])  # 最左的點
            rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])  # 最右的點
            middle = tuple((np.array(leftmost) + np.array(rightmost)) // 2)  # 利用最左和最右 找出圓心
            if count in answer:  # 再來比對myAnswer和answer 看myAnswer的五個答案 一個一個比對看有沒有在answer的list裡面
                cv.circle(image, middle, 45, (0, 255, 0), -1)  # 答對 綠色
            else:
                cv.circle(image, middle, 45, (0, 0, 255), -1)  # 答錯 紅色
        count += 1


def get_grade(img):
    """取得作答分數"""
    myPixel = []  # 存放每個區塊的非空白部分的多寡
    myAnswer = []  # 存放 依據myPixel資料 得到的我的作答資料
    rows = np.vsplit(img, 5)  # 將圖片垂直分割成均等五個小圖片
    for row in rows:
        pixel = []
        cols = np.hsplit(row, 5)  # 再將五個垂直分割的小圖片 進行水平分割成五個均等的小圖片(就變成框出每一個答案所在的位置)
        for col in cols:  # 計算每一row 的每一個col 的非空白部分多寡
            pixel.append(cv.countNonZero(col))
        myPixel.append(pixel)  # 5*5 矩陣
    myPixel = np.array(myPixel)  # numpy序列
    for row in myPixel:
        if row[np.argmax(row)] - row[np.argmin(row)] < 400:  # 若最大值-最小值差異很小 代表這一row 沒有填寫答案 以-1表示
            myAnswer.append(-1)  # 因為答案選項是List的關西 為 0 1 2 3 4, 所以用-1代表沒有作答(也就是空白)
        else:
            myAnswer.append(np.argmax(row))  # 若否,則最大值的index 代表所填寫的答案 也就是有填黑色的部分(非空白部分就會比較多)

    # 更改答案index形式
    for i in range(len(myAnswer)):  # 把五個答案變成 25個編號中的第幾個  再來比對myAnswer 和 answer
        if myAnswer[i] == -1:  # -1 為答案空白 跳過不處理
            continue
        myAnswer[i] = myAnswer[i] + 5 * (i)
    print(myAnswer)
    print(answer)
    grade = len(list(filter(lambda x: x[0] == x[1], zip(answer, myAnswer)))) * 20  # 比對正確答案 答對數目再乘以20
    print(grade)
    return myAnswer, grade


def write_grade(im:imgData,image):
    x, y = im.conArea[2][1][0]  # score欄的矩形框框cnt資料裡的第一個點
    # 寫上分數並根據 分數的位數 調整分數在框框內的位置
    if grade < 10:  # 一位數
        cv.putText(image, str(grade), (x + 75, y + 35), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif grade == 100:  # 三位數
        cv.putText(image, str(grade), (x + 50, y + 35), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:  # 二位數
        cv.putText(image, str(grade), (x + 60, y + 35), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


if __name__ == '__main__':
    # 找三次框 所以四個個物件 攝影機原始畫面 答案卷框 答案框 以及 選項框
    imgCap = imgData()
    imgPaper = imgData()
    imgAnswer = imgData()
    imgAnsSelect = imgData()
    #######################################
    # answer = [3, 3, 2, 4, 1]  # 設定正確答案 D D C E B (A=0,B=1,C=2,D=3,E=4)
    answer = [3, 8, 12, 19, 21]  # 改為對應到25個選項的index  (第一題答案第3個index 第二題答案第3+5個index 第二題答案第2+10個index...)
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)  # 設定攝影機
    #######################################

    while 1:
        success, img = cap.read()
        imgCap.img = img  # 設定攝影機畫面物件的初始圖片
        get_warp(imgCap)  # 取得考卷部分的影像

        imgPaper.img = imgCap.imgWarp  # 設定考卷部分影像 為imgPaper物件的初始圖片
        get_warp(imgPaper)  # 取得答案欄的影像

        # 有可能會因為攝影機的關係 答案欄影像取得失敗 造成程式無法run
        try:
            imgAnswer.img = imgPaper.imgWarp[15:-15, 15:-15]
        except:
            imgAnswer.img = img.copy()

        imgMyAnswer = imgAnswer.img
        imgMyAnswerCopy = imgMyAnswer.copy()

        # 將save改成黑白 並且 二值化
        imgMyAnswerCopy = cv.cvtColor(imgMyAnswerCopy, cv.COLOR_BGR2GRAY)
        imgMyAnswerCopy = cv.GaussianBlur(imgMyAnswerCopy, (5, 5), 1)
        imgMyAnswerCopy = cv.threshold(imgMyAnswerCopy, 185, 255, cv.THRESH_BINARY_INV)[1]

        # 計算分數
        myAnswer, grade = get_grade(imgMyAnswerCopy)

        # 這裡是第三次找框 並重新排列框框順序
        try:
            imgAnsSelect.img = imgMyAnswer
            find_contours(imgAnsSelect)
            myAnswerCon = get_answer_contours(imgAnsSelect.contours, want_Contours_Num=25)
            newCon = reorder_myAnswerCon(myAnswerCon)  # 排序好 新的25個答案圓框框資料

            # 改考卷部分 錯的紅色 對的綠色
            imgResult = np.zeros_like(imgAnswer.img)
            check_Answer(newCon,imgResult)

            imgResult = cv.bitwise_and(imgResult, imgResult, mask=imgMyAnswerCopy)  # 只顯示塗黑的部分
            # 前面try裡面的 list切片 在這裡要補回來
            add_x = np.zeros((15, 600, 3))
            add_y = np.zeros((630, 15, 3))
            imgResult = np.vstack([add_x, imgResult, add_x])
            imgResult = np.hstack([add_y, imgResult, add_y])

            # 第一次 inverse_warp 從答案欄 回到整張考卷
            imgResult = inverse_warp(imgPaper, imgResult, wANDh=True)

            # 寫上分數
            write_grade(imgPaper,imgResult)

            # 第二次 inverse_warp 還原回到攝影機畫面
            imgResult = inverse_warp(imgCap, imgResult, wANDh=False)

            # 更改資料形別 並 將改好的考卷和原攝影機畫面合併
            imgResult = imgResult.astype(np.uint8)
            imgResult = cv.addWeighted(img, 0.8, imgResult, 1, 0)

            # 寫上正確答案
            cv.putText(imgResult, 'Correct Answer: D D C E B', (30, 30), cv.FONT_HERSHEY_SIMPLEX,
                       0.7, (255, 255, 0), 2)

            cv.imshow('imgResult', imgResult)

        except:
            imgResult = img.copy()
            imgWhite = np.full_like(imgResult, 0)
            imgResult = cv.addWeighted(imgResult, 0.8, imgWhite, 1, 0)
            cv.imshow('imgResult', imgResult)

        k = cv.waitKey(50)
        if k == 27:
            break

    cap.release()
    cv.destroyAllWindows()
