import cv2
import numpy as np
import time
import math

width=320
height=240
tile_num_x = 60
tile_num_y = int(tile_num_x*(3/4))
weight_map = np.zeros(tile_num_x*tile_num_y)
weight_img = np.zeros((height,width), np.uint8)
sub_weight = 10
roi = 0
flag = False
#アスペクト比４：３
#タイル内の描画ピクセル数で重み付け、大きなのノイズはタイルのサイズで調整
def main():
    global roi,flag
    tile_list = make_tile_list()
    print(len(tile_list))

    cap = cv2.VideoCapture(0)
    
    cap.set(3, 320)
    cap.set(4, 240)
    width = cap.get(3)
    height = cap.get(4)
    _, frame = cap.read()
    
    tracker = select_tracker()
    tracker_name = str(tracker).split()[0][1:]

    


    while(1):
        

        _, next_frame =  cap.read()
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        diff_frame = fgbg.apply(frame)
        diff_frame = fgbg.apply(next_frame)
        #--白に描画された範囲の中心を求める------------------------
        conditions = (diff_frame[:,:]>254)*1
        #sum_item = np.sum(conditions)
        #if(sum_item != 0):
        #    t_conditions = np.transpose(conditions)
        #    temp = range(len(conditions))
        #    mean_x = np.sum([temp * t_condition for t_condition in t_conditions])//sum_item
        #    temp = range(len(t_conditions))
        #    mean_y = np.sum([temp * condition for condition in conditions])//sum_item
        #    cv2.circle(frame, (mean_y, mean_x),15, (255,0,0), 2)
        ##---------------------------------------------------ri--------
        frame = draw_weight(conditions, tile_list, frame)
       
        cv2.imshow('frame', frame)
        cv2.imshow('diff_frame',diff_frame)
        cv2.imshow('weight_map',weight_img)
        
        frame = next_frame
        
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
        if flag:
            ret = tracker.init(frame, roi)
            break
    #トラック開始--------------------------------
    while True:

        ret, frame = cap.read()

        success, roi = tracker.update(frame)

        (x,y,w,h) = tuple(map(int,roi))

        if success:
            p1 = (x, y)
            p2 = (x+w, y+h)
            cv2.rectangle(frame, p1, p2, (0,255,0), 3)
        else :
            cv2.putText(frame, "Tracking failed!!", (500,400), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)

        # 実行しているトラッカー名を画面に表示
        cv2.putText(frame, tracker_name, (20,600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),3)

        cv2.imshow(tracker_name, frame)

        k = cv2.waitKey(1) & 0xff
        if k == 27 :
            break

    cap.release()
    cv2.destroyAllWindows()
    


def make_tile_list():
    '''
    targetを分割し、そのリストを返します
    tile_numは分割数であり、第一引数がｘ軸方向、第二引数がｙ軸方向
    '''
    
    tile_list = []
    for y in range(0, tile_num_y):
        for x in range(0, tile_num_x):
            tile = (x*width/tile_num_x, y*height/tile_num_y,(x+1)*width/tile_num_x-1, (y+1)*height/tile_num_y-1 )
            tile = tuple(math.floor(i) for i in tile)
            tile_list.append(tile)
    return tile_list

def draw_weight(conditions, tile_list, frame):
    global weight_map
    global flag,roi
    for i in range(tile_num_x*tile_num_y):
        
        
        weight_map[i] += int(np.sum(conditions[tile_list[i][1]:tile_list[i][3]+1,tile_list[i][0]:tile_list[i][2]+1]))
        if weight_map[i]>255:
            #cv2.circle(frame, ((tile_list[i][0]+tile_list[i][2])//2, (tile_list[i][1]+tile_list[i][3])//2),15, (255,0,0), 2)
            
            roi = tuple(tile_list[i])
            flag = True
            weight_map[i]=255
        elif weight_map[i]<0 :
            weight_map[i] = 0
        cv2.rectangle(weight_img, (tile_list[i][0],tile_list[i][1]), (tile_list[i][2],tile_list[i][3]), weight_map[i], thickness=-1)
        #-----------------------------------------------
        #if i%tile_num_x==tile_num_x-1:
        #    print(weight_map[i])
        #else:
        #    print("{},".format(weight_map[i]),end="")

    weight_map -= sub_weight
    return frame
def select_tracker():
    print("Which Tracker API do you use?")
    print("0: Boosting")
    print("1: MIL")
    print("2: KCF")
    print("3: TLD")
    print("4: MedianFlow")
    choice = input("Please select your tracker number: ")

    if choice == '0':
        tracker = cv2.TrackerBoosting_create()
    if choice == '1':
        tracker = cv2.TrackerMIL_create()
    if choice == '2':
        tracker = cv2.TrackerKCF_create()
    if choice == '3':
        tracker = cv2.TrackerTLD_create()
    if choice == '4':
        tracker = cv2.TrackerMedianFlow_create()


    return tracker
if __name__ == "__main__":
    print("start")
    main()

