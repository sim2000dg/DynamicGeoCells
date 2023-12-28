import geopandas as geopd
import pandas as pd
import numpy as np


def init_cells(data_path: str) -> geopd.GeoDataFrame:
    """
    Function to build the GeoPandas DataFrame with the GADM regions (second layer).
    :param data_path: path to the .gpkg file with the GADM data.
    :return: A GeoPandas dataframe with the collected information
    """
    # Read the polygons
    admin = geopd.read_file(data_path, layer="ADM_2")

    # Rename columns
    admin.rename(
        {
            "GID_0": "country_id",
            "GID_1": "region_id",
            "GID_2": "province_id",
            "COUNTRY": "country_name",
            "NAME_1": "region_name",
            "NAME_2": "province_name",
        },
        axis=1,
        inplace=True,
    )

    # subset columns
    admin = admin[
        [
            "country_id",
            "region_id",
            "province_id",
            "country_name",
            "region_name",
            "province_name",
            "geometry",
        ]
    ]

    # handle countries with no provinces
    admin_1 = geopd.read_file(data_path, layer="ADM_1")
    admin_1 = admin_1[~admin_1.GID_0.isin(admin.country_id)]
    admin_1.rename(
        {
            "GID_0": "country_id",
            "GID_1": "region_id",
            "COUNTRY": "country_name",
            "NAME_1": "region_name",
        },
        axis=1,
        inplace=True,
    )
    admin_1.insert(2, "province_id", admin_1.region_id)
    admin_1.insert(5, "province_name", admin_1.region_name)
    admin = pd.concat([admin, admin_1], ignore_index=True, copy=False, join='inner')

    # Get neighbors for each cell
    neighbors_list = []
    for index, cell in admin.iterrows():
        neighbors = admin[~admin.geometry.disjoint(cell.geometry)].index.tolist()
        neighbors.remove(index)
        neighbors_list.append(set(neighbors))
    admin["neighbors"] = neighbors_list

    return admin


def images_in_cells(
    points_csv: str, cells_data: geopd.GeoDataFrame
) -> geopd.GeoDataFrame:
    """
    A Function to compute the number of images contained in each polygon (geocell)
    :param points_csv: The path to the csv file of the sample images
    :param cells_data: The GeoPandas DataFrame with the information on the initial geocells.
    :return: The cells dataframe itself, with two added columns
    """
    # Read the data
    images = pd.read_csv(points_csv)

    # Create a GeoDataFrame from the coordinates of the images
    geometry = geopd.points_from_xy(images.lng, images.lat)
    geo_df = geopd.GeoDataFrame(images.id, crs=4326, geometry=geometry)

    # Perform a spatial join to check if points are within polygons
    points_within_polygons = geopd.sjoin(
        geo_df, cells_data, how="inner", predicate="within"
    )
    # Count the number of points inside each polygon
    count_within_each_polygon = (
        points_within_polygons.groupby("index_right").size().reset_index(name="images")
    )

    # Assign each cell to the respective number of images
    cells = cells_data.merge(
        count_within_each_polygon, left_index=True, right_on="index_right", how="left"
    ).fillna(0)
    cells.rename({"index_right": ""}, axis=1, inplace=True)
    cells.set_index("", inplace=True)

    # Compute the cell area
    cells["area"] = cells_data.to_crs({"proj": "cea"}).area
    # Flag for the merged cells
    cells["Status"] = ["Not Merged" for i in range(len(cells))]

    return cells


def train_test_val(path: str, write: bool):
    """
    Create Train/test/val split
    path: Path to main images.csv file
    write:if you want to save the splits

    """

    df_main = pd.read_csv(path)

    train, val, test = np.split(
        df_main.sample(frac=1, random_state=420),
        [int(0.7 * len(df_main)), int(0.85 * len(df_main))],
    )

    if write:
        pd.DataFrame.to_csv(train, "./train_split.csv", index=False)
        pd.DataFrame.to_csv(val, "./val_split.csv", index=False)
        pd.DataFrame.to_csv(test, "./test_split.csv", index=False)

    return train, val, test


if __name__ == "__main__":
    init_data = init_cells('../gadm_410-levels.gpkg')
    init_data.to_feather('init_cells_out.feather')
