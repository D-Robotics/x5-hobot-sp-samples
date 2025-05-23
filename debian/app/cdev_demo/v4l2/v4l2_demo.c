#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <math.h>
#include <sp_vio.h>
#include <sp_sys.h>
#include <sp_display.h>
#include <argp.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include <linux/videodev2.h>

#define TIMEOUT_S (2) /* 2s */
#define TIMEOUT_US (10000) /* 10 ms */
#define MAX_VIO_FILE_NAME		128

#define BUFFER_COUNT 3
void* v4l2_buffers[BUFFER_COUNT];
int v4l2_buffers_length[BUFFER_COUNT];
void* output_buffers[BUFFER_COUNT];
int output_buffers_length[BUFFER_COUNT];

struct arguments
{
    int width;
    int height;
    int video_id;
    int video_output_id;
    char *input_path;
    int count;
};
static char doc[] = "v4l2_demo";
static struct argp_option options[] = {
    {"width", 'w', "width", 0, "sensor output width"},
    {"height", 'h', "height", 0, "sensor output height"},
    {"video_id", 'n', "id", 0, "the capture video device number"},
    {"video_output_id", 'o', "id", 0, "the output video device number"},
    {"input", 'i', "path", 0, "input file path"},
    {"count", 'c', "number", 0, "capture number"},
    {0}};

static error_t parse_opt(int key, char *arg, struct argp_state *state)
{
    struct arguments *args = state->input;
    switch (key)
    {
        case 'w':
            args->width = atoi(arg);
            break;
        case 'h':
            args->height = atoi(arg);
            break;
        case 'n':
            args->video_id = atoi(arg);
            break;
        case 'o':
            args->video_output_id = atoi(arg);
            break;
        case 'c':
            args->count = atoi(arg);
            break;
        case 'i':
            args->input_path = arg;
            break;
        default:
            return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

int v4l2_device_open(int video_id)
{
	int fd, rc;
    struct v4l2_capability caps;
    char dev_name[64];

    snprintf(dev_name, sizeof(dev_name), "/dev/video%d", video_id);
    fd = open(dev_name, O_RDWR | O_NONBLOCK);
    if (fd < 0) {
        printf("cannot open video device %s\n", dev_name);
        return -1;
    }

    rc = ioctl(fd, VIDIOC_QUERYCAP, &caps);
    if (rc < 0) {
        printf("failed to get device caps for %s (%d=%s)\n", dev_name, errno, strerror(errno));
        close(fd);
        return -1;
    }

    printf("open device: %s (fd=%d)\n", dev_name, fd);
    printf("     driver: %s\n", caps.driver);
    printf("     bus_info: %s\n", caps.bus_info);

    return fd;
}

int v4l2_device_init(int video_fd, int width, int height, int *get_pixelformat)
{
    struct v4l2_capability caps;
    struct v4l2_fmtdesc fmtdesc;
    struct v4l2_frmsizeenum frmsizeenum;
    struct v4l2_format format;
    unsigned int pixelformat = 0;
    int i, j, rc, selected_format = -1;

    if (video_fd < 0)
        return -1;

    memset(&caps, 0, sizeof(caps));
    rc = ioctl(video_fd, VIDIOC_QUERYCAP, &caps);
    if (rc < 0) {
        printf("failed to get device caps(%d=%s)\n", errno, strerror(errno));
        return rc;
    }

    printf("     driver: %s\n", caps.driver);
    printf("       card: %s\n", caps.card);
    printf("    version: %u.%u.%u\n", (caps.version >> 16) & 0xFF, (caps.version >> 8) & 0xFF, (caps.version) & 0xFF);
    printf("   all caps: %08x\n", caps.capabilities);
    printf("device caps: %08x\n", caps.device_caps);

    if (!(caps.capabilities & V4L2_CAP_VIDEO_CAPTURE) &&
        !(caps.capabilities & V4L2_CAP_STREAMING)) {
        printf("streaming capture not supported\n");
        return -1;
    }

    memset(&fmtdesc, 0, sizeof(fmtdesc));
    memset(&frmsizeenum, 0, sizeof(frmsizeenum));
    fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    i = 0;
    while (i < 20) {
        fmtdesc.index = i;
        rc = ioctl(video_fd, VIDIOC_ENUM_FMT, &fmtdesc);
        if (rc < 0) {
            break;
        }

        printf("%2d: %s 0x%08x 0x%x\n", i, fmtdesc.description, fmtdesc.pixelformat, fmtdesc.flags);
        pixelformat = fmtdesc.pixelformat;

        j = 0;
        while (j < 20) {
            frmsizeenum.index        = j;
            frmsizeenum.pixel_format = fmtdesc.pixelformat;
            rc                       = ioctl(video_fd, VIDIOC_ENUM_FRAMESIZES, &frmsizeenum);
            if (rc < 0) {
                break;
            }

            if (frmsizeenum.type == V4L2_FRMSIZE_TYPE_DISCRETE)
                printf("%02d: video_fd %d, width=%d,height=%d\n",  j, video_fd, frmsizeenum.discrete.width,
                     frmsizeenum.discrete.height);
            else
                printf("%02d: video_fd %d, width=[%d %d],height=[%d %d]\n", j, video_fd, frmsizeenum.stepwise.min_width,
                     frmsizeenum.stepwise.max_width, frmsizeenum.stepwise.min_height,
                     frmsizeenum.stepwise.max_height);
            j++;
        }

        i++;
    }

    if(i > 1)
    {
        printf("Please select a format by entering its index: ");
        scanf("%d", &selected_format);

        if (selected_format >= 0 && selected_format < i) 
        {
            fmtdesc.index = selected_format;
            rc = ioctl(video_fd, VIDIOC_ENUM_FMT, &fmtdesc);
            if (rc == 0) 
            {
                printf("You selected format: %s 0x%08x 0x%x\n", fmtdesc.description, fmtdesc.pixelformat, fmtdesc.flags);
                pixelformat = fmtdesc.pixelformat;
            } 
            else 
            {
                perror("Error retrieving selected format");
            }
        } 
        else 
        {
            printf("Invalid format index selected.\n");
        }
    
    }

    if (!pixelformat) {
        printf("video_fd %d, Unsupported pixel format!\n", video_fd);
        return -1;
    }

    memset(&format, 0, sizeof(format));
    format.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    format.fmt.pix.pixelformat = pixelformat;
    format.fmt.pix.width       = width;
    format.fmt.pix.height      = height;
    rc                         = ioctl(video_fd, VIDIOC_S_FMT, &format);
    if (rc < 0) {
        printf("VIDIOC_S_FMT: %s\n",strerror(errno));
        return rc;
    } else if (format.fmt.pix.pixelformat != pixelformat || format.fmt.pix.width != width ||
               format.fmt.pix.height != height) {
        printf("VIDIOC_S_FMT: format (0x%x) / resolution (%dx%d) not supported\n", 
            pixelformat, width, height);
        return -1;
    }

    printf("VIDIOC_S_FMT: format (0x%x) / resolution (%dx%d) preferred\n", 
         format.fmt.pix.pixelformat, format.fmt.pix.width, format.fmt.pix.height);

    *get_pixelformat =  pixelformat;

    return 0;
}

int v4l2_device_reqbufs(int video_fd)
{
    struct v4l2_requestbuffers bufrequest;
    struct v4l2_buffer buffer;
    int i, rc;

    memset(&bufrequest, 0, sizeof(bufrequest));
    bufrequest.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    bufrequest.memory = V4L2_MEMORY_MMAP;
    bufrequest.count  = BUFFER_COUNT;

    rc = ioctl(video_fd, VIDIOC_REQBUFS, &bufrequest);
    if (rc < 0) {
        printf("VIDIOC_REQBUFS: %s\n", strerror(errno));
        return rc;
    }

    for (i = 0; i < BUFFER_COUNT; i++) {
        memset(&buffer, 0, sizeof(buffer));
        buffer.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.index  = i;
        rc            = ioctl(BUFFER_COUNT, VIDIOC_QUERYBUF, &buffer);
        if (rc < 0) {
            printf("VIDIOC_QUERYBUF: %s\n", strerror(errno));
            return rc;
        }
        printf("buffer description:\n");
        printf("offset: %d\n", buffer.m.offset);
        printf("length: %d\n", buffer.length);
        rc = ioctl(video_fd, VIDIOC_QBUF, &buffer);
        if (rc < 0) {
            printf("VIDIOC_QBUF: %s\n", strerror(errno));
            return rc;
        }

        v4l2_buffers[i] =
            mmap(NULL, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, video_fd, buffer.m.offset); 

        if (v4l2_buffers[i] == MAP_FAILED) {
            printf("mmap: %s\n", strerror(errno));
            return -1;
        }

        v4l2_buffers_length[i] = buffer.length;

 		printf("map buffer %p\n", v4l2_buffers[i]);
    }

    return 0;
}

int v4l2_device_query_output_bufs(int video_fd, char *input_path)
{
	int fd, i, rc;
	struct v4l2_buffer buffer;
	struct stat file_stat;
	void *file_data;
    struct v4l2_requestbuffers bufrequest;

	fd = open(input_path, O_RDONLY);
	if (fd < 0) {
		printf("Failed to open file: %s\n", strerror(errno));
		return -1;
	}

	if (fstat(fd, &file_stat) < 0) {
        printf("Failed to get file stats: %s\n", strerror(errno));
        goto err;
    }

	file_data = mmap(NULL, file_stat.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (file_data == MAP_FAILED) {
        printf("Failed to mmap file: %s\n", strerror(errno));
        goto err;
    }

    memset(&bufrequest, 0, sizeof(bufrequest));
    bufrequest.type   = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    bufrequest.memory = V4L2_MEMORY_MMAP;
    bufrequest.count  = BUFFER_COUNT;

    rc = ioctl(video_fd, VIDIOC_REQBUFS, &bufrequest);
    if (rc < 0) {
        printf("VIDIOC_REQBUFS: %s\n", strerror(errno));
        return rc;
    }

	for (i = 0; i < BUFFER_COUNT; i++) {
        memset(&buffer, 0, sizeof(buffer));
        buffer.type   = V4L2_BUF_TYPE_VIDEO_OUTPUT;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.index  = i;
        rc = ioctl(video_fd, VIDIOC_QUERYBUF, &buffer);
        if (rc < 0) {
            printf("VIDIOC_QUERYBUF output buffer: %s\n", strerror(errno));
            goto err;
        }

        printf("output buffer description:\n");
        printf("offset: %d\n", buffer.m.offset);
        printf("length: %d\n", buffer.length);
        rc = ioctl(video_fd, VIDIOC_QBUF, &buffer);
        if (rc < 0) {
            printf("VIDIOC_QBUF: %s\n", strerror(errno));
            return rc;
        }

        output_buffers[i]=
            mmap(NULL, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, video_fd, buffer.m.offset);

        if (output_buffers[i] == MAP_FAILED) {
            printf("mmap: %s\n", strerror(errno));
            goto err;
        }

		if (buffer.length >= file_stat.st_size) {
			memcpy(output_buffers[i], file_data, file_stat.st_size);
		} else {
			printf("buffersize <  picture size!!!, %s\n", strerror(errno));
			goto err;
		}

        output_buffers_length[i] = buffer.length;
    }

err:
	munmap(file_data, file_stat.st_size);
	close(fd);

	return 0;
}

int v4l2_device_reqbufs_free(void)
{
    int i;
    for (i = 0; i < BUFFER_COUNT; i++) {
        munmap(v4l2_buffers[i], v4l2_buffers_length[i]);
        printf("munmap buffer\n");
    }
}

int v4l2_device_reqbufs_output_free(void)
{
    int i;
    for (i = 0; i < BUFFER_COUNT; i++) {
        munmap(output_buffers[i], output_buffers_length[i]);
        printf("munmap buffer\n");
    }
}

int v4l2_device_close(int video_fd)
{
    if (video_fd < 0)
        return -1;

    close(video_fd);
    video_fd = -1;
    return 0;
}

int v4l2_device_start(int video_fd)
{
    int rc;
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    rc = ioctl(video_fd, VIDIOC_STREAMON, &type);
    if (rc < 0) {
        printf("VIDIOC_STREAMON: %s\n", strerror(errno));
    }

    return rc;
}

int v4l2_device_stop(int video_fd)
{
    int rc;
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    rc = ioctl(video_fd, VIDIOC_STREAMOFF, &type);
    if (rc < 0) {
        printf("VIDIOC_STREAMOFF: %s\n", strerror(errno));
    }

    return rc;
}

int v4l2_device_output_start(int video_fd)
{
    int rc;
    int type = V4L2_BUF_TYPE_VIDEO_OUTPUT;

    rc = ioctl(video_fd, VIDIOC_STREAMON, &type);
    if (rc < 0) {
        printf("VIDIOC_STREAMON: %s\n", strerror(errno));
    }

    return rc;
}

int v4l2_device_output_stop(int video_fd)
{
    int rc;
    int type = V4L2_BUF_TYPE_VIDEO_OUTPUT;

    rc = ioctl(video_fd, VIDIOC_STREAMOFF, &type);
    if (rc < 0) {
        printf("VIDIOC_STREAMOFF: %s\n", strerror(errno));
    }

    return rc;
}

void v4l2_cam_send_frame(int video_fd)
{
    struct v4l2_buffer buffer = {0};
    int rc;

    buffer.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;

    rc = ioctl(video_fd, VIDIOC_DQBUF, &buffer);
    if (rc < 0) {
        printf("video_fd %d VIDEO_OUTPUT VIDIOC_DQBUF: %s\n", video_fd, strerror(errno));
        return;
    }
    rc = ioctl(video_fd, VIDIOC_QBUF, &buffer);
    if (rc < 0) {
        printf("video_fd %d VIDEO_OUTPUT VIDIOC_QBUF: %s\n", video_fd, strerror(errno));
    }
}

int dumpToFile(char *srcBuf, unsigned int size,  int width, int height, int pixelformat,int count)
{
    const char *suffix = "nv12";
	FILE *yuvFd = NULL;
	char *buffer = NULL;
    char filename[MAX_VIO_FILE_NAME] = {0};
    int if_2plane = 0;
    unsigned int file_long_size = 0;
    unsigned int uv_start = 0;

    switch (pixelformat) {
        case V4L2_PIX_FMT_SBGGR10:
        case V4L2_PIX_FMT_SGBRG10:
        case V4L2_PIX_FMT_SGRBG10:
        case V4L2_PIX_FMT_SRGGB10:
            suffix = "raw10";
            file_long_size = width * height * 2;
            break;
        case V4L2_PIX_FMT_NV12:
            suffix = "nv12";
            if_2plane = 1;
            file_long_size = width * height * 3 / 2;
			if(size >= file_long_size)
				uv_start = size/3*2;
			else
				uv_start = width * height;
            break;
        default:
            printf("pixelformat(%d) not support\n", pixelformat);
            break;
    }

    snprintf(filename, MAX_VIO_FILE_NAME, "./video_%dx%d_%d.%s", 
        width,
        height,
        count,
        suffix);

	yuvFd = fopen(filename, "w+");

	if (yuvFd == NULL) {
		printf("ERRopen(%s) fail", filename);
		return -1;
	}

	buffer = (char *)malloc(file_long_size);

	if (buffer == NULL) {
		printf(":malloc file");
		fclose(yuvFd);
		return -1;
	}

    if(!if_2plane)
        memcpy(buffer, srcBuf, file_long_size);
    else
    {
        memcpy(buffer, srcBuf, width * height);
	    memcpy(buffer + width*height, srcBuf+uv_start, file_long_size-width*height);
    }

	fflush(stdout);

	fwrite(buffer, 1, file_long_size, yuvFd);

	fflush(yuvFd);

	if (yuvFd)
		fclose(yuvFd);
	if (buffer)
		free(buffer);

	printf("filedump(%s, size(%d) is successed\n", filename, file_long_size);

	return 0;
}

int v4l2_device_dumpframe(int video_fd, int count, int video_output_fd, char *input_path,int width, int height, int pixelformat)
{   
	int rc;
    int run_times = 0;
    int64_t get_buffer_time_ms = 0;
    int64_t get_buffer_time_diff_ms = 0;
    uint64_t buffer_timestamp[2];
    uint64_t buffer_timestamp_diff = 0;
	char file_name[MAX_VIO_FILE_NAME] = {0};
    fd_set fds;

    static struct timeval timeout = {0, TIMEOUT_US};
    struct v4l2_buffer buffer;
    struct timeval get_buffer_time[2] = {0};

    buffer_timestamp[1] = 0;

    if (video_fd < 0)
        return -1;

    buffer.type         = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    do {
        FD_ZERO(&fds);
        FD_SET(video_fd, &fds);
        timeout.tv_sec  = TIMEOUT_S;
        timeout.tv_usec = 0;
		if (!!input_path) 
        {
			v4l2_cam_send_frame(video_output_fd);
        }

        rc = select(video_fd + 1, &fds, NULL, NULL, &timeout);
        if (rc < 0)
            printf("failed to call select: %s\n", strerror(errno));
        else if (!rc)
            printf("select timeout!!!\n");
        else { 
			rc = ioctl(video_fd, VIDIOC_DQBUF, &buffer);
            get_buffer_time[0] = get_buffer_time[1];
            gettimeofday(&get_buffer_time[1], NULL);

        	if (rc < 0) {
            	printf("VIDIOC_DQBUF: %s\n", strerror(errno));
				ioctl(video_fd, VIDIOC_QBUF, &buffer);
            } else if (buffer.flags & V4L2_BUF_FLAG_ERROR) {
                printf("buf flags error, retry dump frame!\n");
				ioctl(video_fd, VIDIOC_QBUF, &buffer);
                count++;
        	} else {
                get_buffer_time_ms = get_buffer_time[1].tv_sec * 1000 + get_buffer_time[1].tv_usec / 1000;
                get_buffer_time_diff_ms = (get_buffer_time[1].tv_sec * 1000 + get_buffer_time[1].tv_usec / 1000) - (get_buffer_time[0].tv_sec * 1000 + get_buffer_time[0].tv_usec / 1000);
                
                buffer_timestamp[0] = buffer_timestamp[1];
                buffer_timestamp[1] = buffer.timestamp.tv_sec * 1000 * 1000 + buffer.timestamp.tv_usec;
                if (buffer_timestamp[0] != 0) {
                    buffer_timestamp_diff = buffer_timestamp[1] - buffer_timestamp[0];
                }
			    printf("video_fd %d: buffer[%d].length = %d, buffer.sequence = %d, buffer.bytesused = %d, buffer.timestamp = %ldms, buffer_timestamp_diff: %ldms, get_buffer_time: %ldms, get_buffer_time_diff: %ldms\n", 
			    	video_fd, buffer.index, buffer.length, buffer.sequence, buffer.bytesused, buffer_timestamp[1] / 1000, buffer_timestamp_diff / 1000, get_buffer_time_ms, get_buffer_time_diff_ms);

				//save frame to file
                dumpToFile((char *)v4l2_buffers[buffer.index], buffer.bytesused, width, height, pixelformat, run_times);

				ioctl(video_fd, VIDIOC_QBUF, &buffer);
                run_times++;
        	}
		}
    } while(count--);

    return 0;
}

static struct argp argp = {options, parse_opt, 0, doc};

int main(int argc, char **argv)
{
    int rc = 0;
    int video_fd = -1;
    int video_output_fd = -1;
    unsigned int pixelformat = 0;

    struct arguments args;
    memset(&args, 0, sizeof(args));
    argp_parse(&argp, argc, argv, 0, 0, &args);
    printf("width:%d, height:%d, video_id:%d, video_output_id:%d, count:%d, input_path:%s\n", 
        args.width, args.height, args.video_id, args.video_output_id, args.count, args.input_path);

    video_fd = v4l2_device_open(args.video_id);
    if (video_fd < 0) {
        printf("open video device failed\n");
        goto err;
    }

    v4l2_device_init(video_fd, args.width, args.height,&pixelformat);
    if (rc < 0)
		goto err0;

    rc = v4l2_device_reqbufs(video_fd);
	if (rc < 0)
		goto err1;

    if (!!args.input_path) {
        if(args.video_id == args.video_output_id)
            video_output_fd = video_fd;
        else
        {
            video_output_fd = v4l2_device_open(args.video_output_id);
            if (video_output_fd < 0) 
            {
                printf("open video output device failed\n");
                goto err1;
            }
        }
        rc = v4l2_device_query_output_bufs(video_output_fd, args.input_path);
        if (rc < 0)
            goto err1;
    }

    if (!!args.input_path)
    {
        rc = v4l2_device_output_start(video_output_fd);
        if (rc < 0)
            goto err1;
    }

    rc = v4l2_device_start(video_fd);
    if (rc < 0)
        goto err1;

    // dqbuf after start stream for 100ms
	usleep(1000 * 100);

    v4l2_device_dumpframe(video_fd, args.count, video_output_fd, args.input_path,args.width, args.height,pixelformat);
    if (rc < 0)
		printf("dump frame failed\n");

    v4l2_device_stop(video_fd);

    if (!!args.input_path)
    {
        rc = v4l2_device_output_stop(video_output_fd);
        if (rc < 0)
            goto err1;
    }

err1:
    v4l2_device_reqbufs_free();
    if (!!args.input_path)
        v4l2_device_reqbufs_output_free();
err0:
    v4l2_device_close(video_output_fd);
    v4l2_device_close(video_fd);
err:
    return -1;

    return 0;
} 