

"""
An example of exporting numpy arrays holding coordiantes of points into csv
"""

import numpy as np
import pandas as pd


__author__ = "Shervin Azadi"
__copyright__ = "???"
__credits__ = ["Shervin Azadi"]
__license__ = "???"
__version__ = "0.0.1"
__maintainer__ = "Shervin Azadi"
__email__ = "shervinazadi93@gmail.com"
__status__ = "Dev"

####################################################
# init an example point cloud
####################################################

point_pos = np.random.rand(100, 3)

# Scale
# point_pos *= np.array([1, 2, 1])
# Translation
# point_pos += np.array([0, 2, 0])

####################################################
# create panda dataframe
####################################################

points_df = pd.DataFrame(
    {'PX': point_pos[:, 0],
     'PY': point_pos[:, 1],
     'PZ': point_pos[:, 2],
     })

####################################################
# save to CSV
####################################################

points_df.to_csv('PY_OUT/point_pos.csv', index=True, float_format='%g')
