import sys
import getopt
from hobot_vio import libsrcampy


def print_usage():
    print("Usage: python3 mipi_camera_scale.py -i <input_path> "
          "-o <output_path> -w <width> -h <height> "
          "[--iheight <iheight>] [--iwidth <iwidth>]")
    sys.exit(2)


def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:o:w:h:",
                                   ["iheight=", "iwidth="])
    except getopt.GetoptError:
        print('Input args error!!!')
        print_usage()

    # Initialize variables
    in_path = out_path = None
    width = height = iheight = iwidth = None

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

    # Validate required arguments
    if not all([in_path, width, height]):
        print("Error: Missing required arguments.")
        print_usage()

    vps = libsrcampy.Camera()
    ret = vps.open_vps(1, 1, iwidth, iheight, width, height)
    if ret:
        print("Error: Failed to open camera.")
        sys.exit(1)

    try:
        with open(in_path, "rb") as fin:
            img = fin.read()
        vps.set_img(img)

        with open(out_path or "output_scale.yuv", "wb+") as fo:
            img = vps.get_img(2, width, height)
            if img is not None:
                fo.write(img)
                print("Encode write image success")
            else:
                print("Encode write image failed")
    finally:
        vps.close_cam()
        print("Test camera scale done!!!")


if __name__ == "__main__":
    main()
