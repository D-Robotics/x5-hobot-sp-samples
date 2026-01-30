import sys
import getopt
from hobot_vio import libsrcampy

# if __name__ == '__main__':
#     vps = libsrcampy.Camera()
#     ret = vps.open_vps(1, 2, 1920, 1080, 640, 480, [300, 300, 1200, 700])
#     print("Camera vps return:%d" % ret)

#     fin = open("input_1080p.yuv", "rb")
#     img = fin.read()
#     fin.close()
#     ret = vps.set_img(img)
#     print ("Process set_img return:%d" % ret)

#     fo = open("output_crop_scale.yuv", "wb+")
#     img = vps.get_img(2, 640, 480)
#     if img is not None:
#         fo.write(img)
#         print("encode write image success")
#     else:
#         print("encode write image failed")
#     fo.close()

#     vps.close_cam()
#     print("test_camera_vps done!!!")


def print_usage():
    print("Usage: python3 mipi_camera_crop_scale.py -i <input_path> "
          "-o <output_path> -w <width> -h <height> --iheight <iheight> "
          "--iwidth <iwidth> -x <crop_x> -y <crop_y> --crop_h <crop_h> "
          "--crop_w <crop_w>")
    sys.exit(2)


def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:o:w:h:x:y:",
                                   ["iheight=", "iwidth=",
                                    "crop_h=", "crop_w="])
    except getopt.GetoptError:
        print('Input args error!!!')
        print_usage()

    # Initialize variables
    in_path = out_path = None
    width = height = iheight = iwidth = \
        crop_x = crop_y = crop_h = crop_w = None

    for opt, arg in opts:
        if opt == "-i":
            in_path = arg
        elif opt == "-o":
            out_path = arg
        elif opt == "-w":
            width = int(arg)
        elif opt == "-h":
            height = int(arg)
        elif opt == "--iheight":
            iheight = int(arg)
        elif opt == "--iwidth":
            iwidth = int(arg)
        elif opt == "-x":
            crop_x = int(arg)
        elif opt == "-y":
            crop_y = int(arg)
        elif opt == "--crop_h":
            crop_h = int(arg)
        elif opt == "--crop_w":
            crop_w = int(arg)

    # Validate required arguments
    if not all([in_path, width, height, iheight, iwidth, crop_x,
                crop_y, crop_h, crop_w]):
        print("Error: Missing required arguments.")
        print_usage()

    vps = libsrcampy.Camera()
    ret = vps.open_vps(1, 2, iwidth, iheight, width, height,
                       [crop_x, crop_y, crop_w, crop_h])
    if ret:
        print("Error: Failed to open camera.")
        sys.exit(1)

    try:
        with open(in_path, "rb") as fin:
            img = fin.read()
        vps.set_img(img)

        with open(out_path or "output_crop_scale.yuv", "wb+") as fo:
            img = vps.get_img(2, width, height)
            if img is not None:
                fo.write(img)
                print("Encode write image success")
            else:
                print("Encode write image failed")
    finally:
        vps.close_cam()
        print("Test camera crop scale done!!!")


if __name__ == "__main__":
    main()
