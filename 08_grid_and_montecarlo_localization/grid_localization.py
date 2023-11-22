""" Grid Localization
    Table 8.1, p 188.
"""

res_x = 0.15    # 0.15m
res_y = 0.15    # 0.15m
res_theta = 5   # degree

max_x = 15
max_y = 15

w_map = int(max_x / res_x)
h_map = int(max_y / res_y)

print("map size: {}, {}".format(w_map, h_map))