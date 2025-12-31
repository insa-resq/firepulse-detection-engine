import argparse
import glob
import os
from pathlib import Path
from typing import TypedDict, LiteralString

import matplotlib.pyplot as plt
import numpy as np
import rasterio

GeoTiffMetadata = TypedDict(
    "GeoTiffMetadata",
    {
        "width": int,
        "height": int,
        "latitude": float,
        "longitude": float
    }
)

def get_geotiff_metadata(image_path: str | Path) -> GeoTiffMetadata:
    """
    Extracts metadata (size, coordinates) from a GeoTIFF.
    """
    with rasterio.open(image_path) as dataset:
        # Get bounds in the image"s native CRS (Coordinate Reference System)
        bounds = dataset.bounds

        # Calculate center point in native CRS
        center_x = (bounds.left + bounds.right) / 2
        center_y = (bounds.bottom + bounds.top) / 2

        # Transform to WGS84 (Latitude/Longitude) if not already
        # EPSG:4326 is the standard for Lat/Lon
        if dataset.crs != "EPSG:4326":
            longitude, latitude = rasterio.warp.transform(
                src_crs=dataset.crs,
                dst_crs=rasterio.CRS.from_epsg("4326"),
                xs=[center_x],
                ys=[center_y]
            )
            return {
                "width": dataset.width,
                "height": dataset.height,
                "latitude": latitude[0],
                "longitude": longitude[0]
            }
        else:
            return {
                "width": dataset.width,
                "height": dataset.height,
                "latitude": center_y,
                "longitude": center_x
            }
        

def read_tiff(file_path: str | bytes | LiteralString | Path):
    with rasterio.open(file_path, "r") as reader:
        profile = reader.profile
        tif_as_array = reader.read()
    return tif_as_array, profile


class DatasetProcessor:
    @staticmethod
    def dataset_generator_seqtoseq(data_path, locations, file_name, label_name, save_path, ts_length=10, interval=3):
        satellite_day = "VIIRS_Day"
        stack_over_location = []
        stack_label_over_locations = []
        n_channels = 8
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for location in locations:
            study_area_path = data_path + "/" + location + "/" + satellite_day + "/"
            file_list = glob.glob(study_area_path + "/*.tif")
            file_list.sort()
            if len(file_list) == 0:
                print("empty file list")
                continue
            array_day, _ = read_tiff(file_list[0])
            array_stack = []
            label_stack = []

            output_shape_x = 256
            output_shape_y = 256
            offset=128

            original_shape_x = array_day.shape[1]
            original_shape_y = array_day.shape[2]

            af_acc_label = np.zeros((output_shape_x, output_shape_y))
            new_base_acc_label = af_acc_label
            file_list_size = len(file_list)
            max_img = np.zeros((n_channels, output_shape_x, output_shape_y), dtype=np.float32)
            for i in range(0, file_list_size, interval):
                if i + ts_length >= file_list_size:
                    i=file_list_size-ts_length
                output_array = np.zeros((ts_length, n_channels, output_shape_x, output_shape_y), dtype=np.float32)
                output_label = np.zeros((ts_length, 3, output_shape_x, output_shape_y), dtype=np.float32)
                for j in range(ts_length):
                    file = file_list[j + i]
                    array_day, _ = read_tiff(file)
                    if os.path.exists(file.replace("VIIRS_Day", "VIIRS_Night")):
                        array_night, _ = read_tiff(file.replace("VIIRS_Day", "VIIRS_Night"))
                        if array_night.shape[0] == 5:
                            print("Day_night miss align")
                            array_night = array_night[3:, :, :]
                        if array_night.shape[0] < 2:
                            print(file.replace("VIIRS_Day", "VIIRS_Night"), "band incomplete")
                            continue
                        if array_night.shape[1] != array_day.shape[1] or array_night.shape[2] != array_day.shape[2]:
                            print("Day Night not match")
                            print(file)
                    else:
                        array_night = np.zeros((2, original_shape_x, original_shape_y))
                    img = np.concatenate((array_day[:6, offset:output_shape_x+offset, offset:output_shape_y+offset], array_night[:, offset:output_shape_x+offset, offset:output_shape_y+offset]), axis=0)
                    img = np.nan_to_num(img[:,:output_shape_x, :output_shape_y])
                    max_img = np.maximum(img, max_img)
                    img = np.concatenate((img[:3,...],img[3:5,...],img[[5],...],img[6:8,...]))
                    if array_day.shape[0]==8:
                        label = np.nan_to_num(array_day[7, :, :], nan=-1)
                    else:
                        label = np.zeros((original_shape_x, original_shape_y))
                    af= array_day[6, :, :]

                    label = np.nan_to_num(label[offset:output_shape_x+offset, offset:output_shape_y+offset])
                    af = np.nan_to_num(af[offset:output_shape_x+offset, offset:output_shape_y+offset])
                    af_acc_label = np.logical_or(af, af_acc_label)
                    if j == interval-1:
                        new_base_acc_label = af_acc_label
                    output_array[j, :n_channels, :, :] = img
                    output_label[j, 0, :, :] = label
                    output_label[j, 1, :, :] = af_acc_label
                    output_label[j, 2, :, :] = af
                af_acc_label = new_base_acc_label
                array_stack.append(output_array)
                label_stack.append(output_label)
            if len(array_stack)==0:
                print("No enough TS")
                continue
            output_array_stacked = np.stack(array_stack, axis=0)
            output_label_stacked = np.stack(label_stack, axis=0)
            stack_over_location.append(output_array_stacked)
            stack_label_over_locations.append(output_label_stacked)
        dataset_stacked_over_locations = np.concatenate(stack_over_location, axis=0).transpose((0,2,1,3,4))
        labels_stacked_over_locations = np.concatenate(stack_label_over_locations, axis=0).transpose((0,2,1,3,4))
        del stack_over_location
        del stack_label_over_locations
        np.save(save_path + "/" + file_name, dataset_stacked_over_locations.astype(np.float32))
        np.save(save_path + "/" + label_name, labels_stacked_over_locations.astype(np.float32))


class TestDatasetProcessor:
    @staticmethod
    def af_test_dataset_generator(location, file_name, save_path, image_size=(256, 256)):
        satellite = "VIIRS_Day"
        ts_length = 10
        n_channels = 8
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        data_path = "data/" + location + "/" + satellite + "/"
        file_list = glob.glob(data_path + "/*.tif")
        file_list.sort()
        if len(file_list) % ts_length != 0:
            num_sequence = len(file_list) // ts_length + 1
        else:
            num_sequence = len(file_list) // ts_length
        array_day, _ = read_tiff(file_list[0])

        array_stack = []
        th = {
            "elephant_hill_fire": [(340, 330), (340, 337), (330, 330), (335, 330), (325, 330), (330, 330), (330, 330),
                                   (330, 330), (340, 330), (335, 330)],
            "eagle_bluff_fire": [(335, 330), (335, 337), (333, 335), (335, 330), (335, 330), (330, 330), (330, 330),
                                 (335, 330), (337, 330), (340, 330)],
            "double_creek_fire": [(360, 330), (340, 337), (340, 325), (335, 330), (337, 330), (337, 330), (335, 335),
                                  (335, 330), (333, 330), (335, 330)],
            "sparks_lake_fire": [(360, 330), (340, 337), (340, 325), (335, 330), (337, 330), (337, 330), (335, 335),
                                 (335, 330), (340, 330), (340, 330)],
            "lytton_fire": [(357, 330), (340, 337), (340, 325), (335, 330), (337, 330), (340, 330), (360, 335),
                            (340, 330), (345, 330), (340, 330)],
            "chuckegg_creek_fire": [(340, 330), (320, 337), (325, 330), (330, 330), (330, 330), (340, 330), (330, 330),
                                    (325, 330), (325, 330), (338, 330)],
            "swedish_fire": [(340, 330), (320, 337), (320, 330), (330, 330), (330, 330), (330, 330), (330, 330),
                             (325, 330), (325, 330), (330, 330)],
            "sydney_fire": [(325, 330), (330, 337), (335, 330), (334, 330), (330, 330), (325, 340), (345, 330),
                            (340, 330), (330, 330), (330, 330)],
            "thomas_fire": [(335, 330), (330, 337), (320, 335), (325, 330), (330, 330), (330, 330), (325, 330),
                            (330, 330), (330, 330), (340, 330)],
            "tubbs_fire": [(340, 330), (325, 337), (330, 330), (320, 330), (325, 330), (325, 330), (330, 330),
                           (330, 330), (350, 350), (320, 330)],
            "carr_fire": [(333, 330), (339, 337), (343, 335), (343, 330), (337, 330), (335, 330), (330, 330),
                          (335, 330), (337, 330), (340, 330)],
            "camp_fire": [(335, 330), (320, 337), (320, 310), (308, 330), (310, 330), (305, 330), (320, 330),
                          (315, 330), (310, 330), (310, 330)],
            "kincade_fire": [(330, 330), (320, 337), (335, 335), (330, 330), (330, 330), (330, 330), (320, 330),
                             (330, 330), (350, 330), (340, 330)],
            "creek_fire": [(355, 330), (340, 337), (335, 335), (330, 330), (330, 330), (330, 330), (335, 330),
                           (340, 330), (337, 330), (340, 330)],
            "blue_ridge_fire": [(330, 330), (325, 337), (350, 335), (340, 330), (335, 330), (330, 330), (330, 330),
                                (335, 330), (337, 330), (340, 330)],
            "dixie_fire": [(340, 330), (335, 337), (345, 345), (340, 330), (345, 360), (340, 330), (333, 330),
                           (335, 330), (340, 350), (345, 350)],
            "mosquito_fire": [(335, 330), (335, 337), (335, 325), (340, 330), (340, 330), (335, 330), (330, 330),
                              (325, 330), (330, 330), (335, 330)],
            "calfcanyon_fire": [(330, 330), (330, 337), (330, 325), (340, 330), (330, 330), (330, 330), (330, 330),
                                (330, 330), (330, 330), (329, 330)]
        }

        th_night = {
            "elephant_hill_fire": [(300, 300), (310, 305), (310, 305), (315, 305), (305, 305), (305, 305), (315, 305),
                                   (305, 305), (305, 305), (315, 305)],
            "eagle_bluff_fire": [(300, 300), (310, 305), (310, 305), (310, 305), (298, 305), (305, 305), (315, 305),
                                 (305, 305), (305, 305), (315, 305)],
            "double_creek_fire": [(310, 300), (305, 305), (310, 305), (310, 305), (298, 305), (305, 305), (308, 305),
                                  (305, 305), (305, 305), (315, 305)],
            "sparks_lake_fire": [(310, 300), (310, 305), (305, 305), (315, 305), (305, 305), (308, 305), (310, 305),
                                 (310, 305), (305, 305), (315, 305)],
            "lytton_fire": [(320, 320), (310, 305), (320, 305), (315, 305), (305, 305), (308, 305), (300, 305),
                            (300, 305), (305, 305), (304, 305)],
            "chuckegg_creek_fire": [(320, 320), (310, 305), (315, 305), (315, 305), (305, 305), (308, 305), (306, 305),
                                    (300, 305), (300, 305), (295, 305)],
            "swedish_fire": [(320, 320), (310, 305), (315, 305), (315, 305), (305, 305), (308, 305), (306, 305),
                             (300, 305), (300, 305), (295, 305)],
            "sydney_fire": [(305, 320), (300, 305), (315, 305), (305, 305), (295, 305), (300, 305), (306, 305),
                            (300, 305), (300, 305), (295, 305)],
            "thomas_fire": [(305, 320), (310, 305), (315, 305), (305, 305), (295, 305), (300, 305), (310, 305),
                            (300, 305), (300, 305), (295, 305)],
            "tubbs_fire": [(305, 320), (310, 305), (315, 305), (305, 305), (295, 305), (300, 305), (315, 305),
                           (300, 305), (300, 305), (300, 305)],
            "carr_fire": [(305, 320), (305, 305), (315, 305), (305, 305), (310, 305), (320, 305), (305, 305),
                          (300, 305), (305, 305), (305, 305)],
            "camp_fire": [(305, 320), (305, 305), (315, 305), (305, 305), (310, 305), (320, 305), (305, 305),
                          (300, 305), (305, 305), (295, 305)],
            "kincade_fire": [(305, 320), (310, 305), (315, 305), (305, 305), (310, 305), (310, 305), (305, 305),
                             (300, 305), (305, 305), (315, 305)],
            "creek_fire": [(305, 320), (320, 305), (320, 305), (315, 305), (310, 305), (310, 305), (305, 305),
                           (300, 305), (305, 305), (315, 305)],
            "blue_ridge_fire": [(305, 320), (320, 305), (320, 305), (315, 305), (310, 305), (310, 305), (305, 305),
                                (300, 305), (310, 305), (315, 305)],
            "dixie_fire": [(300, 320), (320, 305), (320, 305), (315, 305), (310, 305), (310, 305), (305, 305),
                           (300, 305), (310, 305), (315, 305)],
            "mosquito_fire": [(300, 320), (320, 305), (320, 305), (315, 305), (310, 305), (310, 305), (305, 305),
                              (300, 305), (310, 305), (315, 305)],
            "calfcanyon_fire": [(300, 320), (320, 305), (320, 305), (300, 300), (305, 305), (300, 305), (305, 305),
                                (300, 305), (310, 305), (310, 305)]
        }
        for j in range(num_sequence):
            output_array = np.zeros((ts_length, n_channels + 1, image_size[0], image_size[1]))
            if j == num_sequence - 1 and j != 0:
                file_list_size = len(file_list) % ts_length
            else:
                file_list_size = ts_length
            for i in range(file_list_size):
                file = file_list[i + j * 10]
                array_day, profile = read_tiff(file)
                array_night, _ = read_tiff(file.replace("VIIRS_Day", "VIIRS_Night"))
                if os.path.exists(file.replace("VIIRS_Day", "VIIRS_Night")):
                    array_night, _ = read_tiff(file.replace("VIIRS_Day", "VIIRS_Night"))
                else:
                    array_night = np.zeros((2, array_day.shape[1], array_day.shape[2]))
                if array_day.shape[0] != 8:
                    print(file, "band incomplete")
                    continue
                if array_night.shape[0] == 5:
                    print("Day_night miss align")
                    array_night = array_night[3:, :, :]
                if array_night.shape[0] < 2:
                    print(file.replace("VIIRS_Day", "VIIRS_Night"), "band incomplete")
                    continue
                if array_night.shape[1] != array_day.shape[1] or array_night.shape[2] != array_day.shape[2]:
                    print("Day Night not match")
                    print(file)
                    continue

                th_i = th[location]
                th_n = th_night[location]
                if not os.path.exists(f"{save_path}_figure"):
                    os.mkdir(f"{save_path}_figure")
                af = np.zeros(array_day[3, :, :].shape)
                af[:, :] = np.logical_or(array_day[3, :, :] > th_i[i][0], array_day[4, :, :] > th_i[i][1])
                af_img = af
                af_img[np.logical_not(af_img[:, :])] = np.nan
                plt.subplot(221)
                plt.title("day af")
                plt.imshow(array_day[3, :, :])
                plt.imshow(af_img, cmap="hsv", interpolation="nearest")
                plt.subplot(222)
                plt.title("original")
                plt.imshow(array_day[3, :, :])

                af_night = np.zeros(array_night[0, :, :].shape)
                af_night[:, :] = np.logical_or(array_night[0, :, :] > th_n[i][0], array_night[1, :, :] > th_n[i][1])
                af_img_night = af_night
                af_img_night[np.logical_not(af_img_night[:, :])] = np.nan
                plt.subplot(223)
                plt.title("viirs af night")
                plt.imshow(array_night[0, :, :])
                plt.imshow(af_img_night, cmap="hsv", interpolation="nearest")
                plt.subplot(224)
                plt.title("Night original")
                plt.imshow(array_night[0, :, :])
                plt.savefig(f"{save_path}_figure/{location}_{i}.png")
                plt.show()
                plt.close()

                col_start = int(array_day.shape[2] // 2 - 128)
                row_start = int(array_day.shape[1] // 2 - 128)
                array = np.concatenate((array_day[:6, ...], array_night,
                                        np.logical_or(np.nan_to_num(af[np.newaxis, :, :]),
                                                      np.nan_to_num(af_night[np.newaxis, :, :]))))
                plt.imshow(
                    np.logical_or(np.nan_to_num(af[np.newaxis, :, :]), np.nan_to_num(af_night[np.newaxis, :, :]))[
                        0, ...])
                plt.savefig(f"{save_path}_figure/af_label_{location}_{i}.png")
                plt.show()
                array = array[:, row_start:row_start + image_size[0], col_start:col_start + image_size[1]]
                output_array[i, :, :array.shape[1], :array.shape[2]] = np.nan_to_num(array)
            array_stack.append(output_array)

        output_array_stacked = np.stack(array_stack, axis=0)
        np.save(save_path + file_name, output_array_stacked[:, :, :-1, :, :].astype(np.float32))
        np.save(save_path + file_name.replace("img", "label"), output_array_stacked[:, :, -1, :, :].astype(np.float32))

    @staticmethod
    def af_seq_tokenizing_and_test_slicing(location, ts_length, interval, root_path):
        if location in ["val", "train"]:
            root_path = f"{root_path}/dataset_{location}"
            save_path = f"{root_path}/dataset_{location}"
            tokenized_array = np.load(
                os.path.join(root_path, f"af_{location}_img_seqtoseq_alll_{ts_length}i_{interval}.npy")).transpose(
                (0, 3, 4, 2, 1))
            tokenized_label = np.load(
                os.path.join(root_path, f"af_{location}_label_seqtoseq_alll_{ts_length}i_{interval}.npy")).transpose(
                (0, 3, 4, 2, 1))
            tokenized_label = tokenized_label[..., 2]
        else:
            root_path = "/home/z/h/zhao2/CalFireMonitoring/data_train_proj2"
            save_path = "/home/z/h/zhao2/TS-SatFire/dataset/dataset_test"
            tokenized_array = np.load(os.path.join(root_path, f"af_{location}_img.npy")).transpose((0, 3, 4, 1, 2))
            tokenized_label = np.load(os.path.join(root_path, f"af_{location}_label.npy")).transpose((0, 2, 3, 1))
        if tokenized_array.shape[-2] >= ts_length:
            array_concat = []
            label_concat = []
            for i in range(0, tokenized_array.shape[-2], interval):
                if i + ts_length > tokenized_array.shape[-2]:
                    array_concat.append(
                        tokenized_array[:, :, :, tokenized_array.shape[-2] - ts_length:tokenized_array.shape[-2], :])
                    label_concat.append(
                        tokenized_label[:, :, :, tokenized_array.shape[-2] - ts_length:tokenized_array.shape[-2]])
                else:
                    array_concat.append(tokenized_array[:, :, :, i:i + ts_length, :])
                    label_concat.append(tokenized_label[:, :, :, i:i + ts_length])
            tokenized_array = np.concatenate(array_concat, axis=0)
            tokenized_label = np.concatenate(label_concat, axis=0)
        img_array = np.nan_to_num(tokenized_array.transpose((0, 4, 3, 1, 2)))
        img_label = np.nan_to_num(tokenized_label.transpose((0, 3, 1, 2)))
        img_label = np.repeat(img_label[:, np.newaxis, :, :, :], 3, axis=1)
        np.save(os.path.join(save_path, f"af_{location}_img_seqtoseql_{ts_length}i_{interval}.npy"),
                img_array.astype(np.float32))
        np.save(os.path.join(save_path, f"af_{location}_label_seqtoseql_{ts_length}i_{interval}.npy"),
                img_label.astype(np.float32))
            
def main(ts_length, interval, mode):
    df = {}

    val_ids = ["20568194", "20701026", "20562846", "20700973", "24462610", "24462788", "24462753", "24103571",
               "21998313",
               "21751303", "22141596", "21999381", "22712904"]

    df["Id"] = df["Id"].astype(str)
    train_df = df[~df.Id.isin(val_ids)]
    val_df = df[df.Id.isin(val_ids)]

    train_ids = train_df["Id"].values.astype(str)
    val_ids = val_df["Id"].values.astype(str)

    if mode == "train":
        locations = train_ids
    elif mode == "val":
        locations = val_ids
    else:
        locations = ["elephant_hill_fire", "eagle_bluff_fire", "double_creek_fire", "sparks_lake_fire",
                     "lytton_fire",
                     "chuckegg_creek_fire", "swedish_fire", "sydney_fire", "thomas_fire", "tubbs_fire",
                     "carr_fire", "camp_fire", "creek_fire", "blue_ridge_fire", "dixie_fire", "mosquito_fire",
                     "calfcanyon_fire"]
            

    if mode == "train" or mode == "val":
        satimg_processor = DatasetProcessor()
        satimg_processor.dataset_generator_seqtoseq(data_path="/home/z/h/zhao2/CalFireMonitoring/data/",
                                                    locations=locations,
                                                    file_name="af_" + mode + "_img_seqtoseq_alll_" + str(
                                                        ts_length) + "i_" + str(interval) + ".npy",
                                                    label_name="af_" + mode + "_label_seqtoseq_alll_" + str(
                                                        ts_length) + "i_" + str(interval) + ".npy",
                                                    save_path="dataset/dataset_" + mode, ts_length=ts_length,
                                                    interval=interval)
    else:
        for location in locations:
            af_test_processor = TestDatasetProcessor()
            af_test_processor.af_test_dataset_generator(location, save_path="dataset/dataset_test",
                                                        file_name="af_" + location + "_img.npy")
            af_test_processor.af_seq_tokenizing_and_test_slicing(location=location,
                                                                 ts_length=ts_length,
                                                                 interval=interval,
                                                                 root_path="/home/z/h/zhao2/TS-SatFire/dataset")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process GeoTIFF datasets for time-series satellite fire monitoring.")
    
    parser.add_argument("-mode", type=str, help="train/val/test")
    parser.add_argument("-ts", type=int, help="Length of TS")
    parser.add_argument("-it", type=int, help="Interval")
    
    args = parser.parse_args()
    
    main(ts_length=args.ts, interval=args.it, mode=args.mode)
