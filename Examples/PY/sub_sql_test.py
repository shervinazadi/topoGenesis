import sys
import pandas as pd
import sqlite3 as sq


def main(conn):

    result = 12

    return (result)


if __name__ == '__main__':
    # read the argument
    db_path = sys.argv[1]

    # create db connection
    conn = sq.connect(db_path)

    # execute
    result = main(conn)
    print(result)

    # write csv
    # points.to_csv(temp_path + '/points_out.csv')
    # prims.to_csv(temp_path + '/prims_out.csv')
    # detail.to_csv(temp_path + '/detail_out.csv')
