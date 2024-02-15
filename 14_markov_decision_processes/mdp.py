""" Markov Decision Processes
    MDP value iteration and policy (Table 14.1 on page 502)
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

r_min = 0
gamma = 0.97

if __name__ == '__main__':

    map = cv2.imread('mdp_map.png')
    
    map_height = map.shape[0]
    map_width  = map.shape[1]

    # line 2, 3
    value = np.ones([map_height, map_width], dtype=np.float32) * r_min

    x_start = 15
    y_start = 16

    x_dst = 47
    y_dst = 14

    # destination
    value[y_dst, x_dst] = 1.0

    dim_u = 8
    # dx, dy
    u = (
        (-1, -1),
        ( 0, -1),
        (+1, -1),

        (-1, 0),
        (+1, 0),

        (-1, +1),
        ( 0, +1),
        (+1, +1),
    )

    # repeat until convergence
    changed = True
    num_iter = 0
    while changed:
        changed = False

        print('iteration:', num_iter)
        num_iter += 1

        for y in range(map_height):
            for x in range(map_width):
                
                max_next_v = -1e10
                u_opt = -1

                if map[y, x, 0] > 0:
                    # if not wall

                    for idx_u in range(dim_u):
                        dx = u[idx_u][0]
                        dy = u[idx_u][1]

                        x_next = x + dx
                        y_next = y + dy

                        if 0 < y_next < map_height and 0 < x_next < map_width:
                            
                            if map[y_next, x_next, 0] > 0:
                                # if freespace
                                p = 1   # p(x'|u, x)
                                r = -0.01   # penalty for movng
                                v_next = value[y_next, x_next]
                            else:
                                # if wall
                                p = 0   # p(x'|u, x)
                                r = -1  # penalty for collision
                                v_next = 0
                        else:
                            # if out of map
                            p = 0   # p(x'|u, x)
                            r = -1  # penalty for moving out of map
                            v_next = 0

                        next_v = r + v_next*p
                        if max_next_v < next_v:
                            max_next_v = next_v
                            u_opt = idx_u
                    
                
                    if value[y,x] < gamma*max_next_v - 1e-6:
                        value[y,x] = gamma*max_next_v
                        changed = True

    # path planning
    x = x_start
    y = y_start
    path_x = [x]
    path_y = [y]
    while True:

        dx_dst = x - x_dst
        dy_dst = y - y_dst
        r_dst = dx_dst**2 + dy_dst**2

        if r_dst < 1.0:
            break

        max_v = -1e10

        for idx_u in range(dim_u):
            dx = u[idx_u][0]
            dy = u[idx_u][1]

            x2 = x + dx
            y2 = y + dy

            if 0 <= x2 < map_width and 0 <= y2 < map_height:
                if value[y2, x2] > max_v:
                    max_v = value[y2, x2]

                    x_next = x2
                    y_next = y2

        x = x_next
        y = y_next

        path_x.append(x)
        path_y.append(y)

    plt.figure('map')
    plt.imshow(map)
    plt.plot(path_x, path_y, c='r')
    plt.title('map')

    plt.figure('value')
    plt.pcolor(value)
    plt.scatter(x_start, y_start, c='b', marker='x')
    plt.scatter(x_dst, y_dst, c='r', marker='x')
    plt.colorbar()
    plt.axis('equal')
    plt.title('value')


    plt.show()

