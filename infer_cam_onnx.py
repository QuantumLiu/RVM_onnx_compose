import onnxruntime as ort
import numpy as np
import cv2

import time

def main():
    sess = ort.InferenceSession('rvm_mobilenetv3_comp_fp32_hwc.onnx')
    # sess = ort.InferenceSession('rvm_resnet50_comp_fp32_hwc.onnx')

    io = sess.io_binding()

    # Create tensors on CUDA.
    rec = [ ort.OrtValue.ortvalue_from_numpy(np.zeros([1, 1, 1, 1], dtype=np.float32), 'cuda') ] * 4
    downsample_ratio = ort.OrtValue.ortvalue_from_numpy(np.asarray([0.25], dtype=np.float32), 'cuda')

    # Set output binding.
    for name in ['com', 'r1o', 'r2o', 'r3o', 'r4o']:
        io.bind_output(name, 'cuda')


    kernel = np.ones((5,5),np.uint8)
    cap=cv2.VideoCapture(0)


    w=1280
    h=720

    cap.set(3, w)  # width=1920
    cap.set(4, h)  # height=1080

    flag,frame=cap.read()


    def gen_frame(path_video):
        while True:
            cap=cv2.VideoCapture(path_video)
            print("open")
            while True:
                flag,frame=cap.read()
                if flag:
                    yield cv2.resize(frame,(w,h)).astype("float32")#/255.
                    continue
                else:
                    cap.release()
                    break


    baks={}
    for k,i in zip([49,50,51,52],[1,2,3,4]):
        baks[k]=cv2.resize(cv2.imread("backgrounds/bak{}.jpg".format(i)),(w,h)).astype("float32")#/255.

    gen=gen_frame("backgrounds/inception_10s.mp4")

    baks[53]=gen

    bak_bgr=np.ones_like(frame)
    bak_bgr*=np.asarray([120, 255, 155],dtype="uint8")
    bak_bgr=bak_bgr.astype("float32")#/255.

    # Inference loop
    k=-1
    # for i in range(1000):
    while not k==27:
        flag,frame=cap.read()

        start = time.time()


        src=frame[None,:].astype("float32")#/255.
        bak=bak_bgr[None,:]

        
        io.bind_cpu_input('src', src)
        io.bind_cpu_input('bak', bak)
        io.bind_ortvalue_input('r1i', rec[0])
        io.bind_ortvalue_input('r2i', rec[1])
        io.bind_ortvalue_input('r3i', rec[2])
        io.bind_ortvalue_input('r4i', rec[3])
        io.bind_ortvalue_input('downsample_ratio', downsample_ratio)

        sess.run_with_iobinding(io)

        com, *rec = io.get_outputs()
        # for r in rec:
        #     print(r.numpy().shape)
        # fgr=fgr.numpy()[0].transpose(1,2,0)
        # mask=pha.numpy()[0].transpose(1,2,0)
        com=com.numpy()[0]#.transpose(1,2,0)


        # print(mask.shape)

        # mask=cv2.convertScaleAbs(mask*255)

        # mask[mask<0.4]=0.
        # mask[mask>=0.4]=1.
        # ret, binary = cv2.threshold(mask, 90, 255, cv2.THRESH_BINARY)
        # mask=binary.astype("float32")/255
        # mask = cv2.erode(mask,kernel,iterations = 1)
        # mask = cv2.GaussianBlur(mask, (11, 11), 0)
        if k>0:
            print(k)
            bak=baks.get(k,bak_bgr)
            if not isinstance(bak,np.ndarray):
                bak_bgr=next(bak)
            else:
                bak_bgr=bak
        # mask_inv=(1 - mask)

        # foreground = fgr*mask
        # background = bak_bgr*mask_inv

        # com=foreground+background

        # com = fgr * mask + bak_bgr * (1 - mask)
        # cv2.

        # print(fgr.shape,com.shape)
        cv2.imshow("com",com)
        # cv2.imshow("mask",mask)
        end = time.time()
        fps  = 1 / (end-start)
        print( "Estimated frames per second : {0}".format(fps))

        k_n=cv2.waitKey(1)
        k=(k if k_n<0 else k_n)

if __name__ == "__main__":
    # import line_profiler
    # profile = line_profiler.LineProfiler(main)  # 把函数传递到性能分析器
    # profile.enable()  # 开始分析
    main()
    # profile.disable()  # 停止分析
    # profile.print_stats()
