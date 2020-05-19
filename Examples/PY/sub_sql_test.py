import sys
import pandas as pd
import sqlite3 as sq


def main(points, prims, detail):

    points['P_Y'] += 1

    return (points, prims, detail)


if __name__ == '__main__':
    # read the argument
    db_path = sys.argv[1]

    # create db connection
    conn = sq.connect(db_path)

    # create curser
    cursor = conn.cursor()

    # retrieve all data to panda dataframe
    points = pd.read_sql_query("SELECT * FROM POINTS", conn)
    prims = pd.read_sql_query("SELECT * FROM PRIMITIVES", conn)
    detail = pd.read_sql_query("SELECT * FROM DETAIL", conn)

    # execute
    points, prims, detail = main(points, prims, detail)

    # write to db
    points.to_sql(name="POINTS_NEW", con=conn)
    prims.to_sql(name="PRIMITIVES_NEW", con=conn)
    detail.to_sql(name="DETAIL_NEW", con=conn)
