import os
from _paths import nomeroff_net_dir
from nomeroff_net import pipeline
from nomeroff_net.tools import unzip

if __name__ == '__main__':
    number_plate_detection_and_reading = pipeline("number_plate_detection_and_reading", image_loader="opencv")

    images_path = [
        # os.path.join(nomeroff_net_dir, './data/examples/oneline_images/2023-01-05_10-05-33.229450_-_not_detected.jpg'),
        os.path.join(nomeroff_net_dir, './data/examples/oneline_images/E_1_-_M712CC154_0.93_ru_0.99_-_C903EB154_0.93_ru_0.65.jpg'),
        # os.path.join(nomeroff_net_dir, './data/examples/oneline_images/nl_1_-_box_0.78_ru_0.98_-_E625CK154.jpg'),
        # os.path.join(nomeroff_net_dir, './data/examples/oneline_images/nl_2_-_several_1_-_box_0.74_ru_0.90_-_E625CK154.jpg'),
        # os.path.join(nomeroff_net_dir, './data/examples/oneline_images/nl_3_-_box_0.57_ru_0.69_-_E625CK15.jpg'),
        # os.path.join(nomeroff_net_dir, './data/examples/oneline_images/novolugovoe_not_recognized.jpg')
    ]

    for image in images_path:
        result = number_plate_detection_and_reading([image])

        (images, images_bboxs,
        images_points, images_zones, region_ids,
        region_names, count_lines,
        confidences, texts) = unzip(result)

        # (['AC4921CB'], ['RP70012', 'JJF509'])
        # print("$$$", image)
        print("plates:", texts)
        # print("region_ids:", region_ids)
        # print("region_names:", region_names)
        # print("count_lines:", count_lines)
        # print("confidences:", confidences)
