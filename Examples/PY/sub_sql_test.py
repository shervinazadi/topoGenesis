import sys
import pandas as pd


def main(points, prims, detail):

    points += 1

    print(detail)
    return (points, prims, detail)


if __name__ == '__main__':
    # read the argument
    temp_path = sys.argv[1]

    # read csv
    points = pd.read_csv(temp_path + '/points.csv')
    prims = pd.read_csv(temp_path + '/prims.csv')
    detail = pd.read_csv(temp_path + '/detail.csv')

    # execute
    points, prims, detail = main(points, prims, detail)

    # write csv
    points.to_csv(temp_path + '/points_out.csv')
    prims.to_csv(temp_path + '/prims_out.csv')
    detail.to_csv(temp_path + '/detail_out.csv')
