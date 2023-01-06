import open3d as o3d
import open3d.core as o3c
import numpy as np
import cv2
import pickle
import skimage as ski
import copy

import depth_support as ds

# Query available devices
#configs = o3d.t.io.RealSenseSensor.enumerate_devices()

#config = configs[0]

# Open camera using configuration file
#camera = ds.open_camera("rs_d455_config.json")
#param_color = ds.get_color_intrinsics()  # Intrinsic matrix
# Save it for later use
#intrinsics_file = open('intrinsics.pkl', 'wb')
#intrinsics_matrix = copy.deepcopy(param_color.intrinsic_matrix)
#pickle.dump(intrinsics_matrix, intrinsics_file)
# close the file
#intrinsics_file.close()


if __name__ == '__main__':

    # reload for test
    intrinsics_file = open('intrinsics.pkl', 'rb')
    intrinsics_matrix = pickle.load(intrinsics_file)
    # close the file
    intrinsics_file.close()

    # These numbers must match the used calibration grid. The grid is formed by the internal intersection of the chessboard
    square_size = 0.05
    rows = 7
    columns = 10

    # Coordinates of calibration grid points w.r.t. associated frame
    chessboard_points = np.zeros((columns * rows, 3), np.float32)
    chessboard_points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    chessboard_points *= square_size

    # Do we see the calibration plate well?
    color = ski.io.imread('ImageCalib.png')
    #ski.io.imsave("ImageCalib.png", color)
    # color = ski.io.imread("ImageCalib.png")
    gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
    ski.io.imsave("ImageCalib_g.png", gray)
    # gray = ski.io.imread("ImageCalib_g.png")

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (columns, rows), None)

    # Termination criteria for subpixel accuracy corner search algorithm
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Refine corner locations
    corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Compute the chessboard reference frame origin and orientation w.r.t. camera reference frame
    ret, rvec, tvec = cv2.solvePnP(chessboard_points, corners_subpix, intrinsics_matrix, np.zeros(5))

    # Draw and display the corners
    orig_color = color.copy()
    cv2.drawChessboardCorners(color, (columns, rows), corners_subpix, ret)
    cv2.drawFrameAxes(color, intrinsics_matrix, np.zeros(5), rvec, tvec, length=1.0, thickness=3)
    ski.io.imshow(color)
    ski.io.imsave("detected_corners.png", color)

    # SUBSTITUTE HERE THE INDICES USED ON 28/11
    ind_corner = [0, 6, 62, 67]

    # Fill pnt_robt with corner coordinates w.r.t. the robot's EE. Mind the order of the corners!
    # SUBSTITUTE HERE THE COORDINATES FOUND ON 28/11
    pnt_rob = np.array([[151.078, 209.081, -295.966],
                        [451.240, 209.244, -295.829],
                        [251.718, -90.897, -295.829],
                        [501.796, -90.897, -295.879]])
    pnt_rob = pnt_rob / 1000.0
    n_pnt = len(ind_corner)

    pnt_calib = chessboard_points[ind_corner]
    rmat = cv2.Rodrigues(rvec)[0] # rvec is represented as axis-angle. Convert to rotation matrix
    pnt_cam = (rmat @ pnt_calib.T + tvec).T # Turn points into camera reference frame

    # Solve the equation " pnt_rob = s * R * pnt_cam + T " in the least squares sense
    R, T, scale = ds.rigid_registration(pnt_cam, pnt_rob)
    print(f"Pnts cam \n",pnt_cam)
    print(f"Pnts rob \n", pnt_rob)
    print(f"Estimated robot-camera transformation: R, T \n", R, T)

    # Assemble homogeneous transform
    Tr = ds.build_transform_matrix(R, T)

    # Save it for later use
    tr_file = open('cam2rob.pkl', 'wb')
    pickle.dump(Tr, tr_file)
    # close the file
    tr_file.close()

    # reload for test
    tr_file = open('cam2rob.pkl', 'rb')
    Tr_loaded = pickle.load(tr_file)
    # close the file
    tr_file.close()

    ## Test the result

    # Single snap
    bag_reader = o3d.t.io.RSBagReader()
    # Set your path here:
    bag_filename = 'C:\\Users\\light\\robotics_2\\bag_record.bag'
    bag_reader.open(bag_filename)
    # Read one frame to extract metadata, in particular intrinsic matrix
    im_rgbd = bag_reader.next_frame()
    meta = bag_reader.metadata
    depth_intrinsics = o3c.Tensor(meta.intrinsics.intrinsic_matrix)
    h = meta.height
    w = meta.width
    depth_intrinsics_l = o3d.camera.PinholeCameraIntrinsic(w, h, depth_intrinsics.numpy())


    # Convert to legacy format

    # Open3D saves bag files with images in Tensor format. Unfortunately this format does not support all the
    # visualization functions of Open3D, therefore we rebuild the image in the "legacy" format
    color_raw = im_rgbd.color.to_legacy()
    depth_raw = im_rgbd.depth.to_legacy()
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw,
                                                                    convert_rgb_to_intensity=False, depth_trunc=3)

    o3d.visualization.draw_geometries([rgbd_image])

    # Generate point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, depth_intrinsics_l)

    # Create a canonical reference frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame()

    # Copy (assignment operator returns a handler, not a copy!)
    pcd_t = copy.deepcopy(pcd)
    # Apply transformation
    pcd_t.transform(Tr)
    # See the result
    o3d.visualization.draw_geometries([pcd_t, frame])
