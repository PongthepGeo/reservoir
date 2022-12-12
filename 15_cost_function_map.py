#-----------------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
#-----------------------------------------------------------------------------------------#
# pip install plotly
# pip install -U kaleido
#-----------------------------------------------------------------------------------------#

'''
step 1: plot cost function using true color
'''

# cost_function = np.array([
#     					 [0.45, 0.39, 0.77],
#                          [0.69, 0.91, 0.89],
#                          [0.28, 0.65, 0.81]
#                          ])
# fig = plt.figure(figsize=(12, 8))  
# plt.imshow(cost_function, 'Accent')
# # plt.savefig('pictures/' + 'true_color' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
# plt.show()

'''
step 2: plot cost function using smooth contours
'''

fig = go.Figure(data = go.Contour(
        						  z=[[0.45, 0.39, 0.77],
                                     [0.69, 0.91, 0.89],
                                     [0.28, 0.65, 0.81]],
                                      colorscale='Electric',
                                  ))
# fig.write_image('pictures/smooth_color.svg', format='svg')
fig.show()