# By ai_99 (14-07-2022)
import argparse
import cv2
import numpy as np
import sys
import os
import yaml


def parse_args(args):
    parser = argparse.ArgumentParser(description='Dataset visualize script.')
    parser.add_argument('--data-path', help='dataset path', default="./Dataset")
    parser.add_argument('--object-id', help='object id', default=1)
    return parser.parse_args(args)


def main(args=None):

    if args is None:
        args = sys.argv[1:]

    args = parse_args(args)

    object_id = args.object_id
    path = args.data_path

    data_path = os.path.join(path, "data")
    model_path = os.path.join(path, "models")
    image_path = os.path.join(data_path, "{:02d}".format(int(object_id)), "rgb")
    gt_path = os.path.join(data_path, "{:02d}".format(int(object_id)))

    img_list = []
    for i in os.listdir(image_path):
        path1 = os.path.join(image_path, i)
        img_list.append(path1)

    with open(os.path.join(gt_path, "gt.yml")) as stream:
        gt_list = yaml.safe_load(stream)

    with open(os.path.join(gt_path, "info.yml")) as stream:
        info_list = yaml.safe_load(stream)

    yaml_path = os.path.join(model_path, "models_info.yml")

    with open(yaml_path) as fid:
        all_models_dict = yaml.safe_load(fid)

    print("Start 6D ground-truth images")
    run(img_list, gt_list, info_list, all_models_dict, object_id)
    print("Finish")


def get_bbox_3d(model_dict):

    min_point_x = model_dict["min_x"]
    min_point_y = model_dict["min_y"]
    min_point_z = model_dict["min_z"]

    size_x = model_dict["size_x"]
    size_y = model_dict["size_y"]
    size_z = model_dict["size_z"]

    bbox = np.zeros(shape=(8, 3))
    bbox[0, :] = np.array([min_point_x, min_point_y, min_point_z])
    bbox[1, :] = np.array([min_point_x + size_x, min_point_y, min_point_z])
    bbox[2, :] = np.array([min_point_x + size_x, min_point_y + size_y, min_point_z])
    bbox[3, :] = np.array([min_point_x, min_point_y + size_y, min_point_z])
    bbox[4, :] = np.array([min_point_x, min_point_y, min_point_z + size_z])
    bbox[5, :] = np.array([min_point_x + size_x, min_point_y, min_point_z + size_z])
    bbox[6, :] = np.array([min_point_x + size_x, min_point_y + size_y, min_point_z + size_z])
    bbox[7, :] = np.array([min_point_x, min_point_y + size_y, min_point_z + size_z])

    return bbox


def draw_bbox_8_2D(draw_img, bbox_8_2D):

    color = (0, 255, 0)
    thickness = 2

    bbox = np.copy(bbox_8_2D).astype(np.int32)
    bbox = tuple(map(tuple, bbox))

    cv2.line(draw_img, bbox[0], bbox[1], color, thickness)
    cv2.line(draw_img, bbox[1], bbox[2], color, thickness)
    cv2.line(draw_img, bbox[2], bbox[3], color, thickness)
    cv2.line(draw_img, bbox[0], bbox[3], color, thickness)
    cv2.line(draw_img, bbox[4], bbox[5], color, thickness)
    cv2.line(draw_img, bbox[5], bbox[6], color, thickness)
    cv2.line(draw_img, bbox[6], bbox[7], color, thickness)
    cv2.line(draw_img, bbox[4], bbox[7], color, thickness)
    cv2.line(draw_img, bbox[0], bbox[4], color, thickness)
    cv2.line(draw_img, bbox[1], bbox[5], color, thickness)
    cv2.line(draw_img, bbox[2], bbox[6], color, thickness)
    cv2.line(draw_img, bbox[3], bbox[7], color, thickness)

    if len(bbox) == 9:
        cv2.circle(draw_img, bbox[8], 3, color, -1)


def project_bbox_3D_to_2D(points_bbox_3D, rotation_vector, translation_vector, camera_matrix):
    points_bbox_3D = np.concatenate([points_bbox_3D, np.zeros(shape=(1, 3))], axis=0)
    points_bbox_2D, jacobian = cv2.projectPoints(points_bbox_3D, rotation_vector, translation_vector, camera_matrix,
                                                 None)
    points_bbox_2D = np.squeeze(points_bbox_2D)

    return points_bbox_2D


def draw_detections(image, rotations, translations, class_to_bbox_3D, camera_matrix):

    translation_vector = translations
    points_bbox_2D = project_bbox_3D_to_2D(class_to_bbox_3D, rotations, translation_vector, camera_matrix)

    draw_bbox_8_2D(image, points_bbox_2D)

    return image


def transform_rotation(rotation_matrix):
    reshaped_rot_mat = np.reshape(rotation_matrix, newshape=(3, 3))
    return rotation_mat_to_axis_angle(reshaped_rot_mat)


def rotation_mat_to_axis_angle(rotation_matrix):
    axis_angle, jacobian = cv2.Rodrigues(rotation_matrix)

    return np.squeeze(axis_angle)


def run(img_list, gt_list, info_list, all_models_dict, object_id):
    for i in range(len(img_list)):
        print("Showing Image ", "{:04d}.png".format(int(i)))

        image = cv2.imread(img_list[i])

        camera_matrix = np.reshape(info_list[i]["cam_K"], newshape=(3, 3))

        rotations = transform_rotation(gt_list[i][0]["cam_R_m2c"])

        translations = np.array(gt_list[i][0]['cam_t_m2c'])

        class_to_bbox_3D = get_bbox_3d(all_models_dict[int(object_id)])

        image = draw_detections(image, rotations, translations, class_to_bbox_3D, camera_matrix)

        cv2.imshow('Image', image)
        if cv2.waitKey() == ord('q'):
            cv2.destroyAllWindows()
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
