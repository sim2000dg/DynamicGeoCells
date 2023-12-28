import pandas as pd
import shapely
from tqdm import tqdm as tq
import numpy as np
import geopandas as geopd

if __name__ != "__main__":
    from .utility_geo import images_in_cells
from sklearn.cluster import OPTICS
from geovoronoi import voronoi_regions_from_coords
from shapely.set_operations import unary_union, difference
from shapely.validation import make_valid
from sklearn.neighbors import NearestCentroid
from shapely.geometry import Point
from shapely.coordinates import get_coordinates
from operator import itemgetter
import warnings


def merge_cells(
        points_csv: str, cells_data: geopd.GeoDataFrame, threshold: int = 50
) -> geopd.geodataframe:
    """
    The function is designed to construct a GeoPandas DataFrame by merging adjacent geocells,
     based on the number of data points and their respective sizes.
    :param points_csv: The path to the csv file of the sample images
    :param cells_data: The GeoPandas DataFrame with the information on the initial geocells.
    :param threshold: The value of the threshold for the merging
    :return: A GeoPandas DataFrame with the new cell polygons, with number of images and id of images per cell
    """
    cells = images_in_cells(points_csv, cells_data)

    cells_dict = cells.to_dict()

    # A dictionary to preserve all the neighbors for each cell, useful for updating neighbors in constant time
    neigh_id = {idx: set(neighs) for idx, neighs in cells_dict["neighbors"].items()}

    retained_idx = set(
        cells_dict["geometry"].keys()
    )  # A set to store the indexes of all the cells, everytime a cell collapses into another the index is removed
    # from this set
    print("Starting to merge the cells")
    while True:
        if (
                retained_idx.__len__() == 0
        ):  # Stopping condition, there are no longer geocells to be merged
            break

        # Every iteration a set of discarded indexes is maintained, it is not possible updating retained_idx directly
        # because it would change the length of the iterator in the for loop
        discarded_idx = set()
        for idx in tq(
                retained_idx, desc=f"Remaining geocells to merge: {len(retained_idx)}"
        ):
            # Considering only the cells not already merged and with fewer threshold samples
            if (
                    cells_dict["Status"][idx] == "Not Merged"
                    and cells_dict["images"][idx] < threshold
            ):
                all_neigh = list(
                    neigh_id[idx]
                )  # List of the neighbors for the current cell

                # Following the prioritization queue as described in the paper
                neighbors = list(
                    filter(
                        lambda x: cells_dict["images"][x] < threshold
                                  and cells_dict["Status"][x] == "Not Merged"
                                  and cells_dict["country_id"][x] == cells_dict["country_id"][idx]
                                  and cells_dict["region_id"][x] == cells_dict["region_id"][idx],
                        all_neigh,
                    )
                )  # Filtering condition; only cells with fewer than threshold, not already merged,
                # in the same countries and regions are considered

                if len(neighbors) == 0:
                    neighbors = list(
                        filter(
                            lambda x: cells_dict["images"][x] >= threshold
                                      and cells_dict["Status"][x] == "Not Merged"
                                      and cells_dict["country_id"][x]
                                      == cells_dict["country_id"][idx]
                                      and cells_dict["region_id"] == cells_dict["region_id"],
                            all_neigh,
                        )
                    )  # Filtering condition; only cells with more than threshold, not already merged,
                # in the same countries and regions are considered

                if len(neighbors) == 0:
                    neighbors = list(
                        filter(
                            lambda x: cells_dict["images"][x] < threshold
                                      and cells_dict["Status"][x] == "Not Merged"
                                      and cells_dict["country_id"][x]
                                      == cells_dict["country_id"][idx],
                            all_neigh,
                        )
                    )  # Filtering condition; only cells with fewer than threshold, not already merged,
                    # in the same countries, but across regions

                if len(neighbors) == 0:
                    neighbors = list(
                        filter(
                            lambda x: cells_dict["images"][x] >= threshold
                                      and cells_dict["Status"][x] == "Not Merged"
                                      and cells_dict["country_id"][x]
                                      == cells_dict["country_id"][idx],
                            all_neigh,
                        )
                    )  # Filtering condition; only cells with more than threshold, not already merged,
                    # in the same countries, but across regions

                if len(neighbors) > 0:
                    # Get the areas of all the neighbors
                    area_neigh = list(map(cells_dict["area"].get, neighbors))
                    # Selecting the smallest neighbors w.r.t. area
                    smallest_adj = neighbors[np.argmin(area_neigh)]

                    union = cells_dict["geometry"][idx].union(
                        cells_dict["geometry"][smallest_adj]
                    )  # Computing the union of the two cells

                    # Selecting what index to retain based on the area of the cells
                    if cells_dict["area"][idx] >= cells_dict["area"][smallest_adj]:
                        idx_retain = idx
                        idx_discard = smallest_adj
                    else:
                        idx_retain = idx
                        idx_discard = smallest_adj

                    cells_dict["geometry"][
                        idx_retain
                    ] = union  # Assign to the original dataset the new polygon

                    cells_dict["area"][idx_retain] = (
                            cells_dict["area"][idx_retain] + cells_dict["area"][idx_discard]
                    )  # Assign the new area size, as the sum of the two areas, to the cell related to the index to
                    # be retained

                    cells_dict["images"][idx_retain] = (
                            cells_dict["images"][idx_retain]
                            + cells_dict["images"][idx_discard]
                    )  # The number of images of the two united cells are obviously the sum of thw two values

                    cells_dict["Status"][
                        idx_discard
                    ] = "Merged"  # The cell flag w.r.t. the index discarded is set to Merged

                    neigh_id[idx_retain] = neigh_id[idx_retain].union(
                        neigh_id[idx_discard]
                    )  # union of the neighbors

                    # Drop the idx discarded from all its neighbors
                    [
                        neigh_id[adjacent].remove(idx_discard)
                        for adjacent in neigh_id[idx_discard]
                        if idx_discard in neigh_id[adjacent]
                    ]

                    # Add the merged cell to all the neighbors of the discarded one
                    [neigh_id[idx].add(idx_retain) for idx in neigh_id[idx_discard]]

                    # Drop the idx retained from the neighbors of the new geocell
                    neigh_id[idx_retain].remove(idx_retain) if idx_retain in neigh_id[
                        idx_retain
                    ] else None

                    del neigh_id[
                        idx_discard
                    ]  # Remove the cell index discarded from the dictionary

                    cells_dict["neighbors"][idx_retain] = neigh_id[idx_retain]

                    discarded_idx.add(idx_discard)  # Add the discarded idx to the  set
                else:
                    discarded_idx.add(idx)
            else:
                discarded_idx.add(idx)
        retained_idx.difference_update(
            discarded_idx
        )  # Remove all the discarded indexes at the end of the whole iter
    print("Cells have been merged")
    cells_dict = {
        x: y
        for x, y in cells_dict.items()
        if x not in ["region_id", "region_name", "province_id", "province_name"]
    }

    out_df = geopd.GeoDataFrame(cells_dict, crs=4326)
    out_df = out_df[out_df.Status == "Not Merged"]
    out_df.drop(["Status", "area", "neighbors"], axis=1, inplace=True)

    return out_df


def cluster_split(
        merged_data: geopd.GeoDataFrame,
        points_csv: str,
        minsamples_optics_grid: list[int, int, int],
        xis_optics_grid: list[float, float, float],
        minsize: int,
) -> geopd.GeoDataFrame:
    """
    This function takes care of splitting the (already merged) geocells according to optics clustering together with
    Voronoi tesselation.
    :param merged_data: A GeoPandas DataFrame with the merged cells info; output expected with same signature as output
     of merge_cells.
    :param points_csv: CSV with the info on the points/the images.
    :param minsamples_optics_grid: The grid with the three minsample OPTICS hyperparameter of the OPTICS to try while
     relaxing the algorithm.
    :param xis_optics_grid: The grid with the three xi OPTICS hyperparameters to try while relaxing the algorithm.
    :param minsize: The minsize hyperparameter of the semantic geocells algorithm, an integer.
    :return: A GeoPandas DataFrame with the cells split according to the semantic geocells algorithm.
    """
    if len(minsamples_optics_grid) != len(xis_optics_grid):
        raise ValueError(
            "Both the grids of OPTICS hyperparameters need to have the same size!"
        )

    # Read info on data points
    images_info = pd.read_csv(points_csv)
    geometry_points = [
        Point(xy) for xy in zip(images_info.lng, images_info.lat)
    ]  # Read lat/long of points

    # Build GeoDataFrame for points/images
    geo_images = geopd.GeoDataFrame(images_info.id, crs=4326, geometry=geometry_points)
    # Perform spatial join by checking inclusion of points in geocells
    points_within_polygons = geopd.sjoin(
        geo_images, merged_data, how="inner", predicate="within"
    )
    # For each geocell index, get the indexes of the included images
    idx_images = (
        points_within_polygons.groupby(by="index_right")
        .apply(lambda x: np.array(x.index, dtype=np.int32))
        .reset_index(name="idx_images")
    )

    # Finally, perform inner join and join info on merged cells with the array of image indexes
    merged_cells_img = merged_data.merge(
        idx_images, left_index=True, right_on="index_right", how="left"
    )
    merged_cells_img.rename({"index_right": "cell_idx"}, axis=1, inplace=True)
    merged_cells_img.set_index("cell_idx", inplace=True)

    print("Starting cell splitting...")
    i = 1
    for minsamples, xi in zip(minsamples_optics_grid, xis_optics_grid):
        print(
            f"Starting iteration {i} of {len(minsamples_optics_grid)}, exploring one choice of optics hyperparameters"
        )
        merged_cells_img = optics_routine(
            merged_cells_img, geo_images, minsamples, xi, minsize
        )
        i += 1

    return merged_cells_img


def optics_routine(
        merged_cells_img: geopd.GeoDataFrame,
        point_coords: pd.DataFrame,
        minsamples_optics: int,
        xi_optics: float,
        minsize: int,
) -> geopd.GeoDataFrame:
    """
    The OPTICS routine computes clusters for each geocell based on the points inside them. It eventually splits the
    geocells, creating new ones by computing the Voronoi diagram from the samples within the same cluster.
    This process results in a new GeoPandas DataFrame containing the geometries of the new geocells.
    :param merged_cells_img: The GeoPandas DataFrame linking the merged cells to the
     indexes of the images contained in each of them.
    :param point_coords: The lat/long coordinates of the images.
    :param minsamples_optics: The current minsample hyperparameter for OPTICS.
    :param xi_optics: The current xi hyperparameter for OPTICS.
    :param minsize: The global minsize hyperparameter for the semantic geocell algorithm.
    :return: A new GeoPandas DataFrame with the information on the merged cells for the current optics hyperparams.
    """

    optics_ = OPTICS(
        min_samples=minsamples_optics,
        metric="haversine",
        xi=xi_optics,
        algorithm="ball_tree",
        n_jobs=-1,
    )

    nearest_ = NearestCentroid()

    list_cells = list()  # A list to maintain the rows of the new DataFrame
    accum_errors = 0
    split_number = 0
    # Iterating over the rows of the original DataFrame
    for i, (index, row) in (
            pbar := tq(enumerate(merged_cells_img.iterrows()), total=len(merged_cells_img))
    ):
        pbar.set_description(
            f"Number of cells left to consider: {len(merged_cells_img) - i}, 'New cells built': {split_number}, "
            f"GeomErrors: {accum_errors}"
        )

        if (
                row.images < minsize
        ):  # If the number of the images inside a geocell is fewer than the minsize, the algorithm skips that row
            list_cells.append(row)
        else:
            current_coords = get_coordinates(
                point_coords.loc[row.idx_images].geometry
            )  # Get the coordinates from the GeoSeries

            # Clustering of the points inside the geocell
            unique_coords = np.unique(current_coords, axis=0)  # Handle duplicate points
            labels_cluster = optics_.fit_predict(unique_coords[:, ::-1])

            # Put the labels from the duplicates back in
            labels_cluster_prov = np.zeros(len(current_coords), dtype=np.int64)
            for idx, coords in enumerate(unique_coords):
                labels_cluster_prov[np.all(current_coords == coords, axis=1)] = labels_cluster[idx]
            labels_cluster = labels_cluster_prov

            # Project data onto appropriate UTM projection
            cell_series = geopd.GeoSeries(point_coords.loc[row.idx_images].geometry, crs=4326)
            crs_cell = cell_series.estimate_utm_crs()
            cell_series = cell_series.to_crs(crs_cell)

            # Assign noise to closest clusters
            if len(np.unique(labels_cluster[labels_cluster != -1])) > 1 and np.sum(labels_cluster == -1):
                nearest_.fit(get_coordinates(cell_series[labels_cluster != -1].geometry),
                             labels_cluster[labels_cluster != -1])
                labels_cluster[labels_cluster == -1] = nearest_.predict(
                    get_coordinates(cell_series[labels_cluster == -1].geometry))
            elif ~np.sum(labels_cluster == -1):   # If there is only noise or just one cluster, simply skip
                list_cells.append(row)
                continue

            unique_labels, count_labels = np.unique(
                labels_cluster, return_counts=True
            )  # Clusters labels and number of points per cluster

            tot_points = len(labels_cluster)
            valid_clusters = (
                list()
            )  # A list to maintain the clusters labels that meet the conditions below
            while True:
                mask_considered = ~np.isin(
                    unique_labels, valid_clusters
                )  # Masking as True clusters labels if they are not inside valid_clusters variable

                if np.all(
                        ~mask_considered
                ):  # Check whether we have perfectly covered all the points in the cell
                    break

                count_labels_curr = count_labels[
                    mask_considered
                ]  # Extracting the number of points according to the previous masking
                max_size = np.max(count_labels_curr)

                if (
                        max_size >= minsize and (tot_points - max_size) >= minsize
                ):  # Only clusters that respect these conditions are considered
                    valid_clusters.append(
                        unique_labels[mask_considered][np.argmax(count_labels_curr)]
                    )  # Appending to valid_clusters only the cluster with the highest cardinality
                    tot_points -= max_size
                else:
                    break

            if valid_clusters:
                try:
                    split_number += len(valid_clusters)
                    new_cells = (
                        list()
                    )  # A list to store the new geocells rows for each valid cluster, created from the clustered points

                    bounding_geom = (
                        geopd.GeoSeries(row.geometry, crs=4326)
                        .to_crs(crs_cell)
                        .geometry[0]
                    )  # Parent geometry
                    trans_coords = get_coordinates(
                        cell_series
                    )  # Collecting each single Point object in a single collection

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        vor_reg, vor_dict = voronoi_regions_from_coords(
                            trans_coords, bounding_geom, per_geom=False
                        )  # Computing the Voronoi polygons

                    # Invert the dictionary mapping from Voronoi
                    vor_dict_new = {}
                    # Explicit looping since we may have multiple values from a single key
                    for x, y in vor_dict.items():
                        for value in y:
                            vor_dict_new[value] = x
                    vor_dict = vor_dict_new

                    for idx in valid_clusters:  # Iterating over the valid_clusters
                        # Extracting the indexes of the relevant polygons for the specific cluster
                        region_ids = list(set(itemgetter(*(np.where(labels_cluster == idx)[0]))(
                            vor_dict
                        )))

                        geometry_new = unary_union(
                            itemgetter(*region_ids)(vor_reg)
                        )  # The new geometry is computed as the union of all the geometries
                        # according to the correct cluster label

                        geometry_new = (
                            geopd.GeoSeries(
                                make_valid(geometry_new),
                                crs=crs_cell,
                            )
                            .to_crs(4326)
                            .geometry[0]
                        )
                        # This step converts the geometry to the original UTM

                        n_images = np.sum(
                            labels_cluster == idx
                        )  # Number of images of the new geocell
                        # Indexes of all the images of the new geocell
                        idx_images_cell = row.idx_images[labels_cluster == idx]
                        new_cell_series = pd.Series(
                            [
                                row.country_id,
                                row.country_name,
                                geometry_new,
                                n_images,
                                idx_images_cell,
                            ],
                            index=row.index,
                        )  # Making a new row for the new one geocell of the output DataFrame
                        new_cells.append(new_cell_series)

                    excluded = ~np.isin(
                        labels_cluster, valid_clusters
                    )  # Masking the clustered points as True if they are not inside valid_cluster
                    n_images = np.sum(
                        excluded
                    )  # Number of points in the remainder geometry
                    if n_images > 0:
                        # The residual geocell is computed as the difference between the original one
                        # and the union of the new geometries
                        remainder_geom = make_valid(difference(
                            row.geometry,
                            make_valid(unary_union([x.geometry for x in new_cells])),
                        )).buffer(-1e-5).buffer(1e-5)
                    else:  # if there are no points belonging to the remaining geometry, continue
                        list_cells.extend(new_cells)
                        continue

                    idx_images_cell = row.idx_images[
                        excluded
                    ]  # Extracting the indexes of the images inside of remainder_geom
                    new_cell_series = pd.Series(
                        [
                            row.country_id,
                            row.country_name,
                            remainder_geom,
                            n_images,
                            idx_images_cell,
                        ],
                        index=row.index,
                    )
                    list_cells.extend(
                        new_cells + [new_cell_series]
                    )  # Appending both the new geocells and the residual one to the new DataFrame
                except (shapely.GEOSException, ValueError) as err:
                    print(err)
                    accum_errors += 1
                    list_cells.append(row)
                    split_number -= len(valid_clusters)
                    continue
            else:
                list_cells.append(
                    row
                )  # If valid_clusters is empty, appending the row of the original DataFrame

    out = geopd.GeoDataFrame(list_cells, crs=4326).reset_index(drop=True)
    return out


def geocell_centroid(merged_cells_optics: geopd.GeoDataFrame) -> np.ndarray:
    """
    This function computes the centroid for each geocell. It returns latitude/longitude as a numpy.ndarray.
    :param merged_cells_optics: A GeoPandas DataFrame with the cells after OPTICS info; output expected with same
    signature as output of cluster_split.
    :return: A Numpy array.
    """
    centroids = list()
    for i, (idx, row) in (
            pbar := tq(
                enumerate(merged_cells_optics.iterrows()), total=len(merged_cells_optics)
            )
    ):
        pbar.set_description(
            f"Number of cells left to consider: {len(merged_cells_optics) - i}"
        )
        cell_series = geopd.GeoSeries(row.geometry, crs=4326)
        crs_cell = (
            cell_series.estimate_utm_crs()
        )  # Estimating the current UTM in order to reduce the distortion of the new geocell
        cell_reprojected = cell_series.to_crs(
            crs_cell
        )  # Converting the GeoSeries to the current UTM

        cell_centroid = geopd.GeoSeries(cell_reprojected.geometry.centroid).to_crs(4326)
        centroids.append(cell_centroid.get_coordinates().values)

    centroids = np.concatenate(centroids, axis=0)[:, ::-1]

    return centroids

