import sys
import time
import getopt
from hobot_vio import libsrcampy


def print_usage():
    print("Usage: python3 mipi_camera_streamer.py -w <width> -h <height>")
    sys.exit(2)


def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "w:h:")
    except getopt.GetoptError:
        print('Input args error!!!')
        print_usage()

    width = height = None

    for opt, arg in opts:
        if opt == "-w":
            width = int(arg)
        elif opt == "-h":
            height = int(arg)

    if not all([width, height]):
        print("Error: Missing required arguments.")
        print_usage()

    disp = libsrcampy.Display()
    # For the meaning of parameters, please refer
    # to the relevant documents of HDMI display
    disp.display(0, width, height)

    cam = libsrcampy.Camera()
    ret = cam.open_cam(0, -1, 30, width, height)
    if ret:
        print("Error: Failed to open camera.")
        sys.exit(1)

    ret = libsrcampy.bind(cam, disp)
    print("libsrcampy bind return:%d" % ret)

    time.sleep(10)

    libsrcampy.unbind(cam, disp)
    disp.close()
    cam.close_cam()
    print("Test camera streamer done!!!")


if __name__ == "__main__":
    main()
