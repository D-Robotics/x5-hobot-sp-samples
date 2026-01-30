import sys
import getopt
from hobot_vio import libsrcampy


def print_usage():
    print("Usage: python3 mipi_camera_dump.py -f <fps> -c <count> "
          "-w <width> -h <height>")
    sys.exit(2)


def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "f:c:w:h:")
    except getopt.GetoptError:
        print('Input args error!!!')
        print_usage()

    fps = count = weight = height = None

    for opt, arg in opts:
        if opt == "-f":
            fps = int(arg)
        elif opt == "-c":
            count = int(arg)
        elif opt == "-w":
            weight = int(arg)
        elif opt == "-h":
            height = int(arg)

    # Validate required arguments
    if fps is None or count is None or weight is None or height is None:
        print("Error: All arguments -f, -c, -w, -h are required.")
        print_usage()

    cam = libsrcampy.Camera()
    ret = cam.open_cam(0, -1, fps, weight, height)
    if ret:
        print("Error: Failed to open camera.")
        sys.exit(1)

    for a in range(count):
        img = cam.get_img(1)
        outimg = f'output{a}.yuv'

        try:
            with open(outimg, "wb+") as fo:
                fo.write(img)
            print(f"Write image success, count: {a}")
        except Exception as e:
            print(f"Error writing image {a}: {e}")

    cam.close_cam()
    print("test_camera_dump done!!!")


if __name__ == "__main__":
    main()
