# https://github.com/daQuincy/Bresenham-Algorithm/blob/master/bresenham.py
import numpy as np

def bresenham(x0,y0,x1,y1):
    """
    Bresenham's Line Generation Algorithm
    https://www.youtube.com/watch?v=76gp2IAazV4
    """
    # step 2 calculate difference
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    m = dy/dx
    
    # step 3 perform test to check if pk < 0
    flag = True
    
    line_pixel = []
    line_pixel.append((x0,y0))
    
    step = 1
    if x0>x1 or y0>y1:
        step = -1

    mm = False   
    if m < 1:
        x0, x1 ,y0 ,y1 = y0, y1, x0, x1
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        mm = True
        
    p0 = 2*dx - dy
    x = x0
    y = y0
    
    for i in range(abs(y1-y0)):
        if flag:
            x_previous = x0
            p_previous = p0
            p = p0
            flag = False
        else:
            x_previous = x
            p_previous = p
            
        if p >= 0:
            x = x + step

        p = p_previous + 2*dx -2*dy*(abs(x-x_previous))
        y = y + 1
        
        if mm:
            line_pixel.append([y,x])
        else:
            line_pixel.append([x,y])
            
    line_pixel = np.array(line_pixel).astype(int)
    
    return line_pixel