import cv2
import numpy as np
import time
import math

width=320
height=240
tile_num_x = 40
tile_num_y = int(tile_num_x*(3/4))
weight_map = np.zeros(tile_num_x*tile_num_y)

sub_weight = 5

#アスペクト比４：３
#タイル内の描画ピクセル数で重み付け、大きなのノイズはタイルのサイズで調整
def main():
    tile_list = make_tile_list()

    cap = cv2.VideoCapture(0)
    
    cap.set(3, 320)
    cap.set(4, 240)
    width = int(cap.get(3))
    height = int(cap.get(4))
    _, frame = cap.read()
    
    weight_img = np.zeros((height,width), np.uint8)
    


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
        #-----------------------------------------------------------
        weight_img, frame = draw_weight(conditions, tile_list, weight_img, frame)
        cv2.imshow('frame', frame)
        cv2.imshow('diff_frame',diff_frame)
        cv2.imshow('weight_map',weight_img)
        
        frame = next_frame
        
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    
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

def draw_weight(conditions, tile_list, weight_img, frame):
    global weight_map
    
    for i in range(tile_num_x*tile_num_y):
        
        
        weight_map[i] += int(np.sum(conditions[tile_list[i][1]:tile_list[i][3]+1,tile_list[i][0]:tile_list[i][2]+1]))
        if weight_map[i]>230:
            
            if weight_map[i]>254:
                weight_map=255
        elif weight_map[i]<0 :
            weight_map[i] = 0
        
        cv2.rectangle(weight_img, (tile_list[i][0],tile_list[i][1]), (tile_list[i][2],tile_list[i][3]), weight_map[i], thickness=-1)
        #-----------------------------------------------
        #if i%tile_num_x==tile_num_x-1:
        #    print(weight_map[i])
        #else:
        #    print("{},".format(weight_map[i]),end="")

    weight_map -= sub_weight
    return weight_img, frame
    
if __name__ == "__main__":
    print("start")
    main()

